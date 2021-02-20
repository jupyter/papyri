import builtins
import json
import operator
import os
import random
import shutil
from collections import OrderedDict, defaultdict
from functools import lru_cache
from glob import escape as ge
from pathlib import Path

from flatlatex import converter
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pygments.formatters import HtmlFormatter
from quart_trio import QuartTrio
from there import print

from .config import html_dir, ingest_dir
from .crosslink import IngestedBlobs, RefInfo, find_all_refs, load_one
from .stores import Store
from .graphstore import GraphStore, Key
from .take2 import RefInfo
from .utils import progress


def url(info):
    assert isinstance(info, RefInfo)
    # assume same package/version for now.
    if info.module is None:
        assert info.version is None
        return info.path
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
    gstore = GraphStore(ingest_dir)
    files = store.glob("*/*/module/*.json")
    keys = store.glob((None, None, "module", None))

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
        undefined=StrictUndefined,
    )
    env.globals["isstr"] = lambda x: isinstance(x, str)
    env.globals["len"] = len
    template = env.get_template("root.tpl.j2")
    filenames = [_.name[:-5] for _ in files if _.name.endswith(".json")]
    fns = [k.path for k in keys]
    assert set(fns) == set(filenames)
    tree = {}
    for f in filenames:
        sub = tree
        parts = f.split(".")
        for part in parts:
            if part not in sub:
                sub[part] = {}
            sub = sub[part]

        sub["__link__"] = f

    return template.render(tree=tree)


async def examples(module, store, version, subpath, ext=""):
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
        undefined=StrictUndefined,
    )
    env.globals["len"] = len
    env.globals["url"] = url
    env.globals["unreachable"] = unreachable
    css_data = HtmlFormatter(style="pastie").get_style_defs(".highlight")

    pap_files = store.glob("*/*/papyri.json")
    parts = {module: []}
    for pp in pap_files:
        mod, ver = pp.path.parts[-3:-1]
        parts[module].append((RefInfo(mod, ver, "api", mod), mod))

    efile = store / module / version / "examples" / subpath
    from .take2 import Section

    ex = Section.from_json(json.loads(await efile.read_text()))

    class Doc:
        pass

    doc = Doc()
    doc.logo = None

    return env.get_template("examples.tpl.j2").render(
        pygment_css=css_data,
        module=module,
        parts=parts,
        ext=ext,
        version=version,
        parts_links=defaultdict(lambda: ""),
        doc=doc,
        ex=ex,
    )


async def gallery(module, store, version=None, ext=""):
    if version is None:
        version = "*"

    m = defaultdict(lambda: [])
    print("Gallery will glob:")
    for target_path in store.glob(f"{module}/{version}/module/*.json"):
        data = json.loads(await target_path.read_text())
        data["backrefs"] = []
        i = IngestedBlobs.from_json(data)
        i.process(frozenset(), {})

        for k in [
            u.value for u in i.example_section_data if u.__class__.__name__ == "Fig"
        ]:
            module, v, _, _path = target_path.path.parts[-4:]

            # module, filename, link
            impath = f"/p/{module}/{v}/img/{k}"
            link = f"/p/{module}/{v}/api/{target_path.name[:-5]}"
            name = target_path.name[:-5]
            # figmap.append((impath, link, name)
            m[module].append((impath, link, name))

    for target_path in store.glob(f"{module}/{version}/examples/*"):
        data = json.loads(await target_path.read_text())
        from .take2 import Section

        s = Section.from_json(data)

        for k in [u.value for u in s.children if u.__class__.__name__ == "Fig"]:
            module, v, _, _path = target_path.path.parts[-4:]

            # module, filename, link
            impath = f"/p/{module}/{v}/img/{k}"
            link = f"/p/{module}/{v}/examples/{target_path.name}"
            name = target_path.name
            # figmap.append((impath, link, name)
            m[module].append((impath, link, name))

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
        undefined=StrictUndefined,
    )
    env.globals["len"] = len
    env.globals["url"] = url

    class D:
        pass

    doc = D()
    doc.logo = "logo.png"

    pap_files = store.glob("*/*/papyri.json")
    parts = {module: []}
    for pp in pap_files:
        mod, ver = pp.path.parts[-3:-1]
        parts[module].append((RefInfo(mod, ver, "api", mod), mod))

    return env.get_template("gallery.tpl.j2").render(
        figmap=m,
        pygment_css="",
        module=module,
        parts=parts,
        ext=ext,
        version=version,
        parts_links=defaultdict(lambda: ""),
        doc=doc,
    )


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
                {
                    ".".join(s.split(".")[: i + 1])
                    for s in family
                    if s.startswith(cpath) and "." in s
                },
            )
        )
        siblings[part] = [(s, s.split(".")[-1]) for s in sib]
        cpath += part + "."
    if not siblings["+"]:
        del siblings["+"]
    return siblings


def compute_siblings_II(ref, family):
    parts = ref.split(".") + ["+"]
    siblings = OrderedDict()
    cpath = ""

    # TODO: move this at ingestion time for all the non-top-level.
    for i, part in enumerate(parts):
        candidates = [c for c in family if c.path.startswith(cpath) and "." in c.path]
        # trm down to the right length
        candidates = [
            RefInfo(c.module, c.version, "api", ".".join(c.path.split(".")[: i + 1]))
            for c in candidates
        ]
        sib = list(sorted(set(candidates), key=operator.attrgetter("path")))

        siblings[part] = [(c, c.path.split(".")[-1]) for c in sib]
        cpath += part + "."
    if not siblings["+"]:
        del siblings["+"]
    return siblings


def make_tree(names):

    rd = lambda: defaultdict(rd)
    tree = defaultdict(rd)

    for n in names:
        parts = n.split(".")
        branch = tree
        for p in parts:
            branch = branch[p]
    return tree


def cs2(ref, tree, ref_map):
    parts = ref.split(".") + ["+"]
    siblings = OrderedDict()
    cpath = ""
    branch = tree

    def GET(ref_map, key, cpath):
        if key in ref_map:
            return ref_map[key]
        else:
            # this is a tiny bit weird; and will need a workaround at some
            # point.
            # this happends when one object in the hierarchy has not docs
            # (typically a class which is documented only in __init__)
            # or when a object does not match its qualified name (typically in
            # numpy, and __init__ with:
            #  from .foo import foo
            # leading to xxx.foo meaning being context dependant.
            # for now we return a dummy object.
            # print(f"{key=} seem to be a sibling with")
            # print(f"     {cpath=}, but was not ")
            # print(f"     found when trying to compute navigation for")
            # print(f"     {ref=}")
            # print(f"     Here are possible causes:")
            # print(f"        - it is a class and only __init__ has docstrings")
            # print(f"        - it is stored with the wrong qualname          ")
            # print(f"                                             ")

            return RefInfo("?", "?", "?", key)

    for p in parts:
        res = list(sorted([(f"{cpath}{k}", k) for k in branch.keys() if k != "+"]))
        if res:
            siblings[p] = [
                ##(ref_map.get(c, RINFO("?", "?", "?", c)), c.split(".")[-1])
                (GET(ref_map, c, cpath), c.split(".")[-1])
                for c, k in res
            ]
        else:
            break

        branch = branch[p]
        cpath += p + "."
    return siblings


async def _route(ref, store, version=None, env=None, template=None):
    assert not ref.endswith(".html")
    if env is None:
        env = Environment(
            loader=FileSystemLoader(os.path.dirname(__file__)),
            autoescape=select_autoescape(["html", "tpl.j2"]),
            undefined=StrictUndefined,
        )
        env.globals["len"] = len
        env.globals["url"] = url
    if template is None:
        template = env.get_template("core.tpl.j2")
    if ref == "":
        # root = "*"
        # print("GLOB", f"{root}/*/papyri.json")
        ref = "papyri"
        import papyri

        version = papyri.__version__
    root = ref.split(".")[0]

    papp_files = store.glob(f"{root}/*/papyri.json")
    # TODO: deal with versions
    for p in papp_files:
        aliases = json.loads(await p.read_text())

    known_refs, ref_map = find_all_refs(store)

    # known_refs = []
    # for item in o_family:
    #    module, v = item.path.parts[-4:-2]
    #    known_refs.append(RefInfo(module, v, "api", item.name[:-5]))

    siblings = compute_siblings_II(ref, known_refs)
    # print(siblings)

    # End computing siblings.
    if version is not None:
        files = [store / root / version / "module" / f"{ref}.json"]
    else:
        files = list((store / root).glob(f"*/module/{ge(ref)}.json"))
    if files and await (file_ := files[0]).exists():
        # The reference we are trying to view exists;
        # we will now just render it.
        bytes_ = await file_.read_text()
        assert root is not None
        # assert version is not None
        brpath = store / root / version / "module" / f"{ref}.br"
        print(brpath)
        if await brpath.exists():
            br = await brpath.read_text()
        else:
            br = None
        # known_refs = frozenset(
        #    {str(x.name)[:-5] for x in store.glob("*/*/module/*.json")}
        # )
        env.globals["unreachable"] = unreachable
        # env.globals["unreachable"] = lambda *x: "UNREACHABLELLLLL" + str(x)

        doc_blob = load_one(bytes_, br, known_refs=known_refs, strict=True)
        parts_links = {}
        acc = ""
        for k in siblings.keys():
            acc += k
            parts_links[k] = acc
            acc += "."

        css_data = HtmlFormatter(style="pastie").get_style_defs(".highlight")
        return render_one(
            template=template,
            doc=doc_blob,
            qa=ref,
            ext="",
            parts=siblings,
            parts_links=parts_links,
            backrefs=doc_blob.backrefs,
            pygment_css=css_data,
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


async def img(package, version, subpath=None):
    with open(ingest_dir / package / version / "assets" / subpath, "rb") as f:
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

    store = Store(str(ingest_dir))

    async def r(ref):
        return await _route(ref, store)

    async def full(package, version, sub, ref):
        return await _route(ref, store, version)

    async def full_gallery(module, version):
        return await gallery(module, store, version)

    async def g(module):
        return await gallery(module, store)

    async def gr():
        return await gallery("*", store)

    async def index():
        return await _route("", store)

    async def ex(module, version, subpath):
        return await examples(
            module=module, store=store, version=version, subpath=subpath
        )

    # return await _route(ref, GHStore(Path('.')))

    app.route("/logo.png")(logo)
    app.route("/favicon.ico")(static("favicon.ico"))
    # sub here is likely incorrect
    app.route("/p/<package>/<version>/img/<path:subpath>")(img)
    app.route("/p/<module>/<version>/examples/<path:subpath>")(ex)
    app.route("/p/<module>/<version>/gallery")(full_gallery)
    app.route("/p/<package>/<version>/<sub>/<ref>")(full)
    app.route("/<ref>")(r)
    app.route("/gallery/")(gr)
    app.route("/gallery/<module>")(g)
    app.route("/")(index)
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


@lru_cache
def _ascci_env():
    env = Environment(
        loader=CleanLoader(os.path.dirname(__file__)),
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=StrictUndefined,
    )
    env.globals["len"] = len
    env.globals["unreachable"] = unreachable
    try:

        c = converter()

        def math(s):
            assert isinstance(s, list)
            for x in s:
                assert isinstance(x, str)
            res = [c.convert(_) for _ in s]
            print(res)
            return res

        env.globals["math"] = math
    except ImportError:

        def math(s):
            return s + "($pip install flatlatex for unicode math)"

        env.globals["math"] = math

    template = env.get_template("ascii.tpl.j2")
    return env, template


async def _ascii_render(name, store, known_refs=None, template=None, version=None):
    if store is None:
        store = Store(ingest_dir)
    ref = name
    root = name.split(".")[0]

    if not version:
        version = list((store / root).path.iterdir())[-1].name

    env, template = _ascci_env()
    if known_refs is None:
        known_refs = frozenset({x.name[:-5] for x in store.glob("*/module/*.json")})
    bytes_ = await (store / root / version / "module" / f"{ref}.json").read_text()
    brpath = store / root / version / "module" / f"{ref}.br"
    if await brpath.exists():
        br = await brpath.read_text()
    else:
        br = None

    ## TODO : move this to ingest.
    doc_blob = load_one(bytes_, br, strict=True)
    return render_one(
        template=template,
        doc=doc_blob,
        qa=ref,
        ext="",
        backrefs=doc_blob.backrefs,
        pygment_css=None,
    )


async def ascii_render(name, store=None):

    builtins.print(await _ascii_render(name, store))


async def loc(document: Store, *, store: Store, tree, known_refs, ref_map):
    """
    return data for rendering in the templates

    Parameters
    ----------
    document: Store
        Path the document we need to read and prepare for rendering
    store: Store

        Store into which the document is stored (abstraciton layer over local
        filesystem or a remote store like github, thoug right now only local
        file system works)
    tree:
        tree of object we know about; this will be useful to compute siblings
        for the navigation menu at the top that allow to either drill down the
        hierarchy.
    known_refs: List[RefInfo]
        list of all the reference info for targets, so that we can resolve links
        later on; this is here for now, but shoudl be moved to ingestion at some
        point.
    ref_map: ??
        helper to compute the siblings for agiven hierarchy,

    Returns
    -------
    doc_blob: IngestedBlobs
        document that will be rendered
    qa: str
        fully qualified name of the object we will render
    siblings:
        information to render the navigation dropdown at the top.
    parts_links:
        information to render breadcrumbs with links to parents.


    Notes
    -----

    Note that most of the current logic assume we only have the documentation
    for a single version of a package; when we have multiple version some of
    these heuristics break down.

    """
    if isinstance(document, tuple):
        qa = document.path
        version = document.version
        root = document.module
    else:
        qa = document.name[:-5]
        version = document.path.parts[-3]
        # help to keep ascii bug free.
        # await _ascii_render(qa, store, known_refs=known_refs)
        root = qa.split(".")[0]
    try:
        if isinstance(document, tuple):
            assert isinstance(store, GraphStore)
            bytes_ = store.get(document)
        else:
            bytes_ = await document.read_text()
        if isinstance(store, Store):
            brpath = store / root / version / "module" / f"{qa}.br"
            assert await brpath.exists()
            br = await brpath.read_text()
        elif isinstance(store, GraphStore):
            br = ""
        else:
            assert False
        doc_blob: IngestedBlobs = load_one(
            bytes_, br, known_refs=known_refs, strict=True
        )

    except Exception as e:
        raise RuntimeError(f"error with {document}") from e

    siblings = cs2(qa, tree, ref_map)

    parts_links = {}
    acc = ""
    for k in siblings.keys():
        acc += k
        parts_links[k] = acc
        acc += "."
    try:
        return doc_blob, qa, siblings, parts_links
    except Exception as e:
        raise type(e)(f"Error in {qa}") from e


async def main(ascii, html, dry_run):
    from .graphstore import GraphStore

    gstore = GraphStore(ingest_dir, {})
    store = Store(ingest_dir)
    files = store.glob("*/*/module/*.json")
    gfiles = list(gstore.glob((None, None, "module", None)))

    css_data = HtmlFormatter(style="pastie").get_style_defs(".highlight")
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
        undefined=StrictUndefined,
    )
    env.globals["len"] = len
    env.globals["unreachable"] = unreachable
    env.globals["url"] = url
    template = env.get_template("core.tpl.j2")
    if dry_run:
        output_dir = None
    else:
        output_dir = html_dir / "p"
        output_dir.mkdir(exist_ok=True)
    document: Store

    known_refs, ref_map = find_all_refs(store)
    x_, y_ = find_all_refs(gstore)
    assert x_ == known_refs
    assert y_ == ref_map
    # end

    family = frozenset(_.path for _ in known_refs)

    tree = make_tree(family)

    print("going to erase", html_dir)
    # input("press enter to continue...")
    shutil.rmtree(html_dir)
    random.shuffle(files)
    random.shuffle(gfiles)
    # Gallery
    mv = store.glob("*/*")
    mv2 = gstore.glob((None, None))
    assert set((m, v) for (m, v) in mv2) == set(
        (item.path.parent.name, item.path.name) for item in mv
    )
    for item in mv:
        version, module = item.path.name, item.path.parent.name
        data = await gallery(module, store, version, ext=".html")
        (output_dir / module / version / "gallery").mkdir(parents=True, exist_ok=True)
        with (output_dir / module / version / "gallery" / "index.html").open("w") as f:
            f.write(data)

    for p, key in progress(gfiles, description="Rendering..."):
        module, v = key.module, key.version
        if ascii:
            qa = key.path
            await _ascii_render(qa, store, family, version=v)
        if html:
            doc_blob, qa, siblings, parts_links = await loc(
                key,
                store=gstore,
                tree=tree,
                known_refs=known_refs,
                ref_map=ref_map,
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
            if not dry_run:
                (output_dir / module / v / "api").mkdir(parents=True, exist_ok=True)
                with (output_dir / module / v / "api" / f"{qa}.html").open("w") as f:
                    f.write(data)

    key = Key("papyri", "0.0.2", "module", "papyri")

    module, v = "papyri", "0.0.2"
    if html:
        doc_blob, qa, siblings, parts_links = await loc(
            key,
            store=gstore,
            tree=tree,
            known_refs=known_refs,
            ref_map=ref_map,
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
        if not dry_run:
            with (html_dir / "index.html").open("w") as f:
                f.write(data)

    if not dry_run:
        assets = store.glob("*/*/assets/*")
        for asset in assets:
            module, version, _, _name = asset.parts[-4:]
            b = html_dir / "p" / module / version / "img"
            b.mkdir(parents=True, exist_ok=True)

            shutil.copy(asset.path, b / asset.name)
