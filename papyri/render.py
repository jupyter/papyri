import json
import os
from collections import defaultdict
from pathlib import Path
from there import print

from jinja2 import Environment, FileSystemLoader, select_autoescape, StrictUndefined
from quart_trio import QuartTrio

from .config import html_dir, ingest_dir
from .crosslink import (
    load_one,
    resolve_,
    IngestedBlobs,
    paragraph,
    paragraphs,
    P2,
)
from .stores import BaseStore, GHStore, Store
from .take2 import (
    Lines,
    Paragraph,
    make_block_3,
    Link,
    Node,
    Section,
    BlockDirective,
    DefListItem,
    Example,
    BlockVerbatim,
)
from .utils import progress
from collections import OrderedDict

from typing import List


from dataclasses import dataclass


@dataclass
class RefInfo:
    module: str
    version: str
    kind: str
    path: str


def url(info):
    assert isinstance(info, RefInfo)
    return f"/p/{info.module}/{info.version}/api/{info.path}"


def unreachable(*obj):
    assert False, f"Unreachable: {obj=}"

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




def root():
    store = Store(ingest_dir)
    files = store.glob("*/*/module/*.json")

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
        undefined=StrictUndefined,
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
    for p in store.glob(f"{module}/*/module/*.json"):
        data = json.loads(await p.read_text())
        i = IngestedBlobs.from_json(data)

        for k in {u[1] for u in i.example_section_data if u[0] == "fig"}:
            figmap.append((p.parts[-3], k, p.name[:-5]))

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
        undefined=StrictUndefined,
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph

    return env.get_template("gallery.tpl.j2").render(figmap=figmap)


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

def compute_siblings(ref, family):
    parts = ref.split(".") + ["+"]
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
    return siblings


def make_tree(names):    
    from collections import defaultdict, OrderedDict
    rd = lambda: defaultdict(rd)
    tree = defaultdict(rd)

    for n in names:
        parts = n.split('.')
        branch = tree
        for p in parts:
            branch = branch[p]
    return tree

def cs2(ref, tree):
    parts = ref.split('.')+["+"]
    siblings = OrderedDict()
    cpath = ""
    branch = tree
    for p in parts:
        res = list(sorted([(f"{cpath}{k}",k) for k in branch.keys() if k != '+']))
        if res:
            siblings[p] = res
        else:
            break
        
        branch = branch[p]
        cpath += p + "."
    return siblings

        
from pygments.formatters import HtmlFormatter


async def _route(ref, store, version=None):
    assert isinstance(store, BaseStore)
    assert ref != "favicon.ico"
    assert not ref.endswith(".html")

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
        undefined=StrictUndefined,
    )
    env.globals["paragraph"] = paragraph
    env.globals["len"] = len
    env.globals["url"] = url

    template = env.get_template("core.tpl.j2")

    root = ref.split(".")[0]

    papp_files = store.glob(f"{root}/*/papyri.json")
    # TODO: deal with versions
    for p in papp_files:
        aliases = json.loads(await p.read_text())


    family = sorted(list(store.glob("*/*/module/*.json")))
    family = [str(f.name)[:-5] for f in family]

    siblings = compute_siblings(ref, family)

    # End computing siblings.
    if version is not None:
        files = [store / root / version / "module" / f"{ref}.json"]
    else:
        from glob import escape as ge

        files = list((store / root).glob(f"*/module/{ge(ref)}.json"))
    if files and await (file_ := files[0]).exists():
        # The reference we are trying to view exists;
        # we will now just render it.
        bytes_ = await file_.read_text()
        brpath = store / root / "module" / f"{ref}.br"
        if await brpath.exists():
            br = await brpath.read_text()
        else:
            br = None
        all_known_refs = frozenset(
            {str(x.name)[:-5] for x in store.glob("*/*/module/*.json")}
        )
        env.globals["unreachable"] = unreachable
        # env.globals["unreachable"] = lambda *x: "UNREACHABLELLLLL" + str(x)

        doc_blob = load_one(bytes_, br, qa=ref)
        parts_links = {}
        acc = ""
        for k in siblings.keys():
            acc += k
            parts_links[k] = acc
            acc += "."
        prepare_doc(doc_blob, ref, all_known_refs)
        return render_one(
            template=template,
            doc=doc_blob,
            qa=ref,
            ext="",
            parts=siblings,
            parts_links=parts_links,
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
            str(s.name)[:-5] for s in store.glob(f"{r}/*/module/{ref}*.json")
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
        return error.render(backrefs=list(set(br)), tree=tree, ref=ref, module=root)


async def img(subpath):
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

    async def full_img(package, version, sub, subpath):
        return await img(subpath)

    async def full(package, version, sub, ref):
        return await _route(ref, Store(str(ingest_dir)), version)

    async def g(module):
        return await gallery(module, Store(str(ingest_dir)))

    async def gr():
        return await gallery("*", Store(str(ingest_dir)))

    # return await _route(ref, GHStore(Path('.')))

    app.route("/logo.png")(logo)
    app.route("/favicon.ico")(static("favicon.ico"))
    # sub here is likely incorrect
    app.route("/p/<package>/<version>/<sub>/img/<path:subpath>")(full_img)
    app.route("/p/<package>/<version>/<sub>/<ref>")(full)
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
    template,
    doc: IngestedBlobs,
    qa,
    ext,
    *,
    backrefs,
    pygment_css=None,
    parts={},
    parts_links={},
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


    try:
        return template.render(
            doc=doc,
            qa=qa,
            version=doc.version,
            module=qa.split(".")[0],
            backrefs=backrefs,
            ext=ext,
            parts=parts,
            parts_links=parts_links,
            pygment_css=pygment_css,
        )
    except Exception as e:
        raise ValueError("qa=", qa) from e


from functools import lru_cache


@lru_cache
def _ascci_env():
    env = Environment(
        loader=CleanLoader(os.path.dirname(__file__)),
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=StrictUndefined,
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    env.globals["unreachable"] = unreachable
    template = env.get_template("ascii.tpl.j2")
    return env, template


async def _ascii_render(name, store, known_refs=None, template=None):
    if store is None:
        store = Store(ingest_dir)
    ref = name
    root = name.split(".")[0]

    env, template = _ascci_env()
    if known_refs is None:
        known_refs = frozenset({x.name[:-5] for x in store.glob("*/module/*.json")})
    bytes_ = await (store / root / "module" / f"{ref}.json").read_text()
    brpath = store / root / "module" / f"{ref}.br"
    if await brpath.exists():
        br = await brpath.read_text()
    else:
        br = None

    ## TODO : move this to ingest.
    doc_blob = load_one(bytes_, br, qa=name)
    try:
        prepare_doc(doc_blob, ref, known_refs)
    except Exception as e:
        raise type(e)(f"Error preparing ASCII {name}")
    return render_one(
        template=template,
        doc=doc_blob,
        qa=ref,
        ext="",
        backrefs=doc_blob.backrefs,
        pygment_css=None,
    )


async def ascii_render(name, store=None):
    print(await _ascii_render(name, store))


from .crosslink import TreeReplacer, DirectiveVisiter

def prepare_doc(doc_blob, qa, known_refs):
    assert hash(known_refs)
    sections_ = [
        "Parameters",
        "Returns",
        "Raises",
        "Yields",
        "Attributes",
        "Other Parameters",
    ]

    local_refs = []
    for s in sections_:
        local_refs = local_refs + [x[0] for x in doc_blob.content[s] if x[0]]

    ### dive into the example data, reconstruct the initial code, parse it with pygments,
    # and append the highlighting class as the third element
    # I'm thinking the linking strides should be stored separately as the code
    # it might be simpler, and more compact.
    # TODO : move this to ingest.
    visitor = DirectiveVisiter(qa, known_refs, local_refs)

    doc_blob.example_section_data = visitor.visit(doc_blob.example_section_data)

    # doc_blob.example_section_data = processed_example_data_nonlocal(
    #    doc_blob.example_section_data, known_refs, qa=qa
    # )

    # partial lift of paragraph parsing....
    # TODO: Move this higher in the ingest

    doc_blob.refs = [
        (resolve_(qa, known_refs, local_refs, x), x) for x in doc_blob.refs
    ]

    for section in ["Extended Summary", "Summary", "Notes"] + sections_:
        if section in doc_blob.content:
            doc_blob.content[section] = visitor.visit(doc_blob.content[section])
        else:
            assert False

    for d in doc_blob.see_also:
        new_desc = []
        for dsc in d.descriptions:
            new_desc.append(visitor.visit(dsc))
            visitor.local = []
            visitor.total = []

        d.descriptions = new_desc


async def loc(document, *, store, tree, known_refs):
    qa = document.name[:-5]
    # help to keep ascii bug free.
    # await _ascii_render(qa, store, known_refs=known_refs)
    root = qa.split(".")[0]
    try:
        bytes_ = await document.read_text()
        brpath = store / root / "module" / f"{qa}.br"
        if await brpath.exists():
            br = await brpath.read_text()
        else:
            br = None
        doc_blob: IngestedBlobs = load_one(bytes_, br, qa=qa)

    except Exception as e:
        raise RuntimeError(f"error with {document}") from e

    siblings = cs2(qa, tree)

    parts_links = {}
    acc = ""
    for k in siblings.keys():
        acc += k
        parts_links[k] = acc
        acc += "."
    try:
        for in_out in doc_blob.example_section_data:
            if in_out.__class__.__name__ == "text":
                for it in in_out[0]:
                    assert not isinstance(it, tuple), it

        prepare_doc(doc_blob, qa, known_refs)
        return doc_blob, qa, siblings, parts_links
    except Exception as e:
        raise type(e)(f"Error in {qa}") from e


async def main(ascii, html):
    store = Store(ingest_dir)
    files = store.glob("*/module/*.json")
    css_data = HtmlFormatter(style="pastie").get_style_defs(".highlight")
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
        undefined=StrictUndefined,
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    template = env.get_template("core.tpl.j2")

    known_refs = frozenset({x.name[:-5] for x in store.glob("*/*/module/*.json")})
    assert len(known_refs) >= 1

    html_dir.mkdir(exist_ok=True)
    document: Store
    family = sorted(list(store.glob("*/module/*.json")))
    family = [str(f.name)[:-5] for f in family]
    assert set(family) == set(known_refs)
    family = known_refs

    tree = make_tree(known_refs)
    env.globals["unreachable"] = unreachable

    for p, document in progress(files, description="Rendering..."):
        if (
            document.name.startswith("__")
            or not document.name.endswith(".json")
            or document.name.endswith("__papyri__.json")
        ):
            assert False, document.name
        if ascii:
            qa = document.name[:-5]
            await _ascii_render(qa, store, known_refs)
        if html:
            doc_blob, qa, siblings, parts_links = await loc(
                document,
                store=store,
                tree=tree,
                known_refs=known_refs,
            )
            data = render_one(
                template=template,
                doc=doc_blob,
                qa=qa,
                ext=".html",
                parts=siblings,
                parts_links=parts_links,
                backrefs=doc_blob.backrefs,
                pygment_css=css_data,
            )
            with (html_dir / f"{qa}.html").open("w") as f:
                f.write(data)

    assets = store.glob("*/assets/*")
    for asset in assets:
        b = html_dir / "img" / asset.parts[-3] / asset.parts[-2]
        b.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(asset.path, b / asset.name)
