import json
import os
from collections import defaultdict
from pathlib import Path
from there import print

from jinja2 import Environment, FileSystemLoader, select_autoescape
from quart_trio import QuartTrio

from .config import html_dir, ingest_dir
from .crosslink import load_one, resolve_, IngestedBlobs, paragraph, paragraphs, P2
from .stores import BaseStore, GHStore, Store
from .take2 import Lines, Paragraph, make_block_3
from .utils import progress


class CleanLoader(FileSystemLoader):
    """
    A loader for ascii/ansi that remove all leading spaces and pipes  until the last pipe.
    """

    def get_source(self, *args, **kwargs):
        (source, filename, uptodate) = super().get_source(*args, **kwargs)
        return until_ruler(source), filename, uptodate


def until_ruler(doc):
    """
    Utilities to clean jinja template;

    Remove all ``|`` and `` `` until the last leading ``|``

    """
    lines = doc.split("\n")
    new = []
    for l in lines:

        while len(l.lstrip()) >= 1 and l.lstrip()[0] == "|":
            l = l.lstrip()[1:]
        new.append(l)
    return "\n".join(new)


from pygments import lex
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


def get_classes(code):
    list(lex(code, PythonLexer()))
    FMT = HtmlFormatter()
    classes = [FMT.ttype2class.get(x) for x, y in lex(code, PythonLexer())]
    classes = [c if c is not None else "" for c in classes]
    return classes


def root():
    store = Store(ingest_dir)
    files = store.glob("*/module/*.json")

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["isstr"] = lambda x: isinstance(x, str)
    env.globals["len"] = len
    template = env.get_template("root.tpl.j2")
    filenames = [_.name[:-5] for _ in files if _.name.endswith(".json")]
    tree = {}
    for f in filenames:
        sub = tree
        parts = f.split(".")
        for i, part in enumerate(parts):
            if part not in sub:
                sub[part] = {}
            sub = sub[part]

        sub["__link__"] = f

    return template.render(tree=tree)


async def gallery(module, store):

    from pathlib import Path
    import json

    from papyri.crosslink import IngestedBlobs

    figmap = []
    for p in store.glob(f"{module}/module/*.json"):
        data = json.loads(await p.read_text())
        i = IngestedBlobs.from_json(data)

        for k in {u[1] for u in i.example_section_data if u[0] == "fig"}:
            figmap.append((p.parts[-3], k, p.name[:-5]))

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    env.globals["len"] = len

    return env.get_template("gallery.tpl.j2").render(figmap=figmap)


async def _route(ref, store):
    assert isinstance(store, BaseStore)
    assert ref != "favicon.ico"
    assert not ref.endswith(".html")

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    env.globals["len"] = len

    template = env.get_template("core.tpl.j2")

    root = ref.split(".")[0]

    papp_files = store.glob(f"{root}/papyri.json")
    for p in papp_files:
        aliases = json.loads(await p.read_text())

    # here we compute the siblings at each level; as well as one level down
    # this is far from efficient and a hack, but it helps with navigation.
    # I'm pretty sure we load the full library while we could
    # load only the current module id not less, and that this could
    # be done at ingest time or cached.
    # So basically in the breadcrumps
    # IPython.lib.display.+
    #  - IPython will be siblings with numpy; scipy, dask, ....
    #  - lib (or "IPython.lib"), with "core", "display", "terminal"...
    #  etc.
    #  - + are deeper children's
    #
    # This is also likely a bit wrong; as I'm sure we want to only show
    # submodules or sibling modules and not attribute/instance of current class,
    # though that would need loading the files and looking at the types of
    # things. likely want to store that in a tree somewhere But I thing this is
    # doable after purely as frontend thing.

    family = sorted(list(store.glob("*/module/*.json")))
    family = [str(f.name)[:-5] for f in family]
    parts = ref.split(".") + ["+"]
    from collections import OrderedDict

    siblings = OrderedDict()
    cpath = ""
    # TODO: move this at ingestion time for all the non-top-level.
    for i, part in enumerate(parts):
        sib = list(
            sorted(
                set(
                    [
                        ".".join(s.split(".")[: i + 1])
                        for s in family
                        if s.startswith(cpath) and "." in s
                    ]
                )
            )
        )
        siblings[part] = [(s, s.split(".")[-1]) for s in sib]
        cpath += part + "."
    if not siblings["+"]:
        del siblings["+"]

    # End computing siblings.

    file_ = store / root / "module" / f"{ref}.json"
    if await file_.exists():
        # The reference we are trying to view exists;
        # we will now just render it.
        bytes_ = await file_.read_text()
        brpath = store / root / "module" / f"{ref}.br"
        if await brpath.exists():
            br = await brpath.read_text()
        else:
            br = None
        doc_blob = load_one(bytes_, br)
        local_refs = [x[0] for x in doc_blob.content["Parameters"] if x[0]] + [
            x[0] for x in doc_blob.content["Returns"] if x[0]
        ]
        all_known_refs = [str(x.name)[:-5] for x in store.glob("*/module/*.json")]
        env.globals["resolve"] = resolve_(ref, all_known_refs, local_refs)

        ### dive into the example data, reconstruct the initial code, parse it with pygments,
        # and append the highlighting class as the third element
        # I'm thinking the linking strides should be stored separately as the code
        # it might be simpler, and more compact.
        for i, (type_, (in_out)) in enumerate(doc_blob.example_section_data):
            if type_ == "code":
                if len(in_out) == 2:
                    in_, out = in_out
                    in_out.append("old_version")
                elif len(in_out) == 3:
                    in_, out, ce_status = in_out
                classes = get_classes("".join([x for x, y in in_]))
                for ii, cc in zip(in_, classes):
                    # TODO: Warning here we mutate objects.
                    ii.append(cc)

        return render_one(
            template=template,
            doc=doc_blob,
            qa=ref,
            ext="",
            parts=siblings,
            backrefs=doc_blob.backrefs,
            pygment_css=HtmlFormatter(style="pastie").get_style_defs(".highlight"),
        )
    else:
        # The reference we are trying to render does not exists
        # just try to have a nice  error page and try to find local reference and
        # use the phantom file to list the backreferences to this.
        # it migt be a page, or a module we do not have documentation about.
        r = ref.split(".")[0]
        this_module_known_refs = [
            str(s.name)[:-5] for s in store.glob(f"{r}/module/{ref}*.json")
        ]
        brpath = store / "__phantom__" / f"{ref}.json"
        if await brpath.exists():
            br = json.loads(await brpath.read_text())
        else:
            br = []

        # compute a tree from all the references we have to have a nice browsing
        # interfaces.
        tree = {}
        for f in this_module_known_refs:
            sub = tree
            parts = f.split(".")[len(ref.split(".")) :]
            for i, part in enumerate(parts):
                if part not in sub:
                    sub[part] = {}
                sub = sub[part]

            sub["__link__"] = f

        error = env.get_template("404.tpl.j2")
        return error.render(backrefs=list(set(br)), tree=tree, ref=ref)


def img(subpath):
    with open(ingest_dir / subpath, "rb") as f:
        return f.read()


def static(name):
    here = Path(os.path.dirname(__file__))

    def f():
        return (here / name).read_bytes()

    return f


def logo():

    path = os.path.abspath(__file__)
    dir_path = Path(os.path.dirname(path))
    with open((dir_path / "papyri-logo.png"), "rb") as f:
        return f.read()


def serve():

    app = QuartTrio(__name__)

    async def r(ref):
        return await _route(ref, Store(str(ingest_dir)))

    async def g(module):
        return await gallery(module, Store(str(ingest_dir)))

    async def gr():
        return await gallery("*", Store(str(ingest_dir)))

    # return await _route(ref, GHStore(Path('.')))

    app.route("/logo.png")(logo)
    app.route("/favicon.ico")(static("favicon.ico"))
    app.route("/<ref>")(r)
    app.route("/img/<path:subpath>")(img)
    app.route("/gallery/")(gr)
    app.route("/gallery/<module>")(g)
    app.route("/")(root)
    port = os.environ.get("PORT", 5000)
    print("Seen config port ", port)
    prod = os.environ.get("PROD", None)
    if prod:
        app.run(port=port, host="0.0.0.0")
    else:
        app.run(port=port)


def render_one(
    template, doc: IngestedBlobs, qa, ext, *, backrefs, pygment_css=None, parts={}
):
    """
    Return the rendering of one document

    Parameters
    ----------
    template
        a Jinja@ template object used to render.
    doc : DocBlob
        a Doc object with the informations for current obj
    qa : str
        fully qualified name for current object
    ext : str
        file extension for url  – should likely be removed and be set on the template
        I think that might be passed down to resolve maybe ?
    backrefs : list of str
        backreferences of document pointing to this.
    parts : Dict[str, list[(str, str)]
        used for navigation and for parts of the breakcrumbs to have navigation to siblings.
        This is not directly related to current object.

    """
    # TODO : move this to ingest likely.
    # Here if we have too many references we group them on where they come from.
    if len(backrefs) > 30:

        b2 = defaultdict(lambda: [])
        for ref in backrefs:
            mod, _ = ref.split(".", maxsplit=1)
            b2[mod].append(ref)
        backrefs = (None, b2)
    else:
        backrefs = (backrefs, None)

    # partial lift of paragraph parsing....
    # TODO: Move this higher in the ingest
    sections_ = [
        "Parameters",
        "Returns",
        "Raises",
        "Yields",
        "Attributes",
        "Other Parameters",
    ]
    for s in sections_:
        for i, p in enumerate(doc.content[s]):
            if p[2]:
                doc.content[s][i] = (p[0], p[1], paragraphs(p[2]))

    for s in ["Extended Summary", "Summary", "Notes"]:
        if s in doc.content:
            data = doc.content[s]
            res = []
            for it in P2(data):
                res.append((it.__class__.__name__, it))
            doc.content[s] = res

    for d in doc.see_also:
        assert isinstance(d.descriptions, list), qa
        d.descriptions = paragraphs(d.descriptions)
    try:
        return template.render(
            doc=doc,
            qa=qa,
            version=doc.version,
            module=qa.split(".")[0],
            backrefs=backrefs,
            ext=ext,
            parts=parts,
            pygment_css=pygment_css,
        )
    except Exception as e:
        raise ValueError("qa=", qa) from e


async def _ascii_render(name, store=None):
    if store is None:
        store = Store(ingest_dir)
    ref = name
    root = name.split(".")[0]

    env = Environment(
        loader=CleanLoader(os.path.dirname(__file__)),
        lstrip_blocks=True,
        trim_blocks=True,
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    template = env.get_template("ascii.tpl.j2")

    known_refs = [x.name[:-5] for x in store.glob("*/module/*.json")]
    bytes_ = await (store / root / "module" / f"{ref}.json").read_text()
    brpath = store / root / "module" / f"{ref}.br"
    if await brpath.exists():
        br = await brpath.read_text()
    else:
        br = None
    blob = load_one(bytes_, br)

    # TODO : move this to ingest.
    local_ref = [x[0] for x in blob.content["Parameters"] if x[0]] + [
        x[0] for x in blob.content["Returns"] if x[0]
    ]

    # TODO : move this to ingest.
    env.globals["resolve"] = resolve_(ref, known_refs, local_ref)
    for i, (type_, in_out) in enumerate(blob.example_section_data):
        if type_ == "code":
            if len(in_out) == 2:
                in_, out = in_out
            elif len(in_out) == 3:
                in_, out, _ = in_out
            else:
                raise ValueError
            for ii in in_:
                ii.append(None)

    return render_one(
        template=template,
        doc=blob,
        qa=ref,
        ext="",
        backrefs=blob.backrefs,
        pygment_css=None,
    )


async def ascii_render(name, store=None):
    print(await _ascii_render(name, store))


async def main():
    store = Store(ingest_dir)
    files = store.glob("*/module/*.json")

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    template = env.get_template("core.tpl.j2")

    known_refs = [x.name[:-5] for x in store.glob("*/module/*.json")]

    html_dir.mkdir(exist_ok=True)
    document: Store
    for p, document in progress(files, description="Rendering..."):
        if (
            document.name.startswith("__")
            or not document.name.endswith(".json")
            or document.name.endswith("__papyri__.json")
        ):
            assert False, document.name
        qa = document.name[:-5]
        root = qa.split(".")[0]
        try:
            bytes_ = await document.read_text()
            brpath = store / root / "module" / f"{qa}.br"
            if await brpath.exists():
                br = await brpath.read_text()
            else:
                br = None
            doc_blob: IngestedBlobs = load_one(bytes_, br)

        except Exception as e:
            raise RuntimeError(f"error with {document}") from e

        # for p,(qa, doc_blob:IngestedBlobs) in progress(nvisited_items.items(), description='Rendering'):
        local_refs = [x[0] for x in doc_blob.content["Parameters"] if x[0]] + [
            x[0] for x in doc_blob.content["Returns"] if x[0]
        ]
        env.globals["resolve"] = resolve_(qa, known_refs, local_refs)
        for i, (type_, in_out) in enumerate(doc_blob.example_section_data):
            if type_ == "code":
                if len(in_out) == 2:
                    in_, out = in_out
                    in_out.append("old_version")
                elif len(in_out) == 3:
                    in_, out, ce_status = in_out
                for ii in in_:
                    ii.append(None)

        with (html_dir / f"{qa}.html").open("w") as f:
            try:
                f.write(
                    render_one(
                        template=template,
                        doc=doc_blob,
                        qa=qa,
                        ext=".html",
                        backrefs=doc_blob.backrefs,
                        pygment_css=None,
                    )
                )
            except Exception as e:
                raise ValueError(f"error writin {qa=}") from e
