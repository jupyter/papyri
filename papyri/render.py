import logging
import json
import math
import operator
import os
import random
import shutil
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    select_autoescape,
    Template,
)
from pygments.formatters import HtmlFormatter
from quart import Response, redirect, send_from_directory
from quart_trio import QuartTrio
from rich.logging import RichHandler

try:
    import minify_html
except ModuleNotFoundError:
    minify_html = None  # type: ignore[assignment]

from . import config as default_config
from . import take2
from .config import ingest_dir
from .crosslink import IngestedBlobs, find_all_refs
from .graphstore import GraphStore, Key
from .myst_ast import (
    MLink,
    MList,
    MListItem,
    MRoot,
    MText,
    MHeading,
    MParagraph,
    MRoot,
    MImage,
)
from .take2 import (
    RefInfo,
    Section,
    encoder,
    Link,
    SeeAlsoItem,
    DefList,
    DefListItem,
    Fig,
    MUnimpl,
)
from .tree import TreeReplacer, TreeVisitor
from .utils import dummy_progress, progress

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("papyri")

CSS_DATA = HtmlFormatter(style="pastie").get_style_defs(".highlight")


def backrefs_to_myst(backrefs: List[RefInfo], LR: "LinkReifier") -> List[Any]:
    """
    Take a list of backreferences and group them by the module they are comming from,
    then turn this into a myst list.
    """

    if not backrefs:
        return []

    group = defaultdict(lambda: [])
    for ref in backrefs:
        assert isinstance(ref, RefInfo)
        if "." in ref.path:
            mod, _ = ref.path.split(".", maxsplit=1)
        else:
            mod = ref.path
        [link] = LR.replace_RefInfo(ref)
        assert isinstance(link, MLink), link
        group[mod].append(link)

    children: Any = [MHeading(children=[MText("Backreferences")], depth=3)]
    for mod, links in sorted(group.items()):
        children.append(MHeading(children=[MText(f"From {mod}")], depth=4))
        lchild = []
        for link in links:
            lchild.append(MListItem(children=[LR.visit(link)], spread=False))
        children.append(MList(children=lchild, ordered=False, start=1, spread=False))
    return children


def minify(s: str) -> str:
    return minify_html.minify(
        s, minify_js=True, remove_processing_instructions=True, keep_closing_tags=True
    )


def unimplemented(*obj):
    print("Unimplemtned:", str(obj))
    return str(obj)


def unreachable(*obj):
    # return str(obj)
    assert False, f"Unreachable: {obj=}"


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


def compute_siblings_II(
    ref: str, family: frozenset[RefInfo]
) -> Dict[str, List[Tuple[RefInfo, str]]]:
    """
    Compute the full tree of siblings for all the items.


    Dictionary of the type:

    numpy:

        [(RefFor numpy.array, "array"),(RefFor sin, "sin"),....]

    core:
        [(RefFor numpy.core.masked_array, "masked_array", ...)
    where:
        ...


    """
    assert isinstance(ref, str)

    module_versions = defaultdict(lambda: set())
    for f in family:
        module_versions[f.module].add(f.version)

    module_versions_max = {k: max(v) for k, v in module_versions.items()}  # type: ignore [type-var]

    family = frozenset(
        {f for f in family if f.version == module_versions_max[f.module]}
    )

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


def make_tree(names: Iterable[str]):
    rd = lambda: defaultdict(rd)  # type: ignore
    tree = defaultdict(rd)  # type: ignore

    for n in names:
        parts = n.split(".")
        branch = tree
        for p in parts:
            branch = branch[p]
    return tree


def cs2(ref, tree, ref_map):
    """
    IIRC this is quite similar to compute_siblings(_II),
    but more efficient as we know we are going to compute
    all the siblings and not just the local one when rendering a single page.

    """
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
        res = list(sorted((f"{cpath}{k}", k) for k in branch.keys() if k != "+"))
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


def _uuid():
    return uuid.uuid4().hex


class HtmlRenderer:
    def __init__(self, store: GraphStore, *, sidebar, prefix, trailing_html):
        assert prefix.startswith("/")
        assert prefix.endswith("/")
        self.progress = progress
        self.store = store
        self.env = Environment(
            loader=FileSystemLoader(Path(os.path.dirname(__file__)) / "templates"),
            autoescape=select_autoescape(["html", "tpl.j2"]),
            undefined=StrictUndefined,
        )

        self.env.trim_blocks = True
        self.env.lstrip_blocks = True
        self.prefix = prefix
        extension = ".html" if trailing_html else ""
        self.resolver = Resolver(store, prefix, extension)
        self.LR = LinkReifier(resolver=self.resolver)
        assert hasattr(self.LR, "_replacements"), self.LR
        self.env.globals["len"] = len
        self.env.globals["url"] = self.resolver.resolve
        self.env.globals["unimplemented"] = unimplemented
        self.env.globals["unreachable"] = unreachable
        self.env.globals["sidebar"] = sidebar
        self.env.globals["dothtml"] = extension
        self.env.globals["uuid"] = _uuid

    def compute_graph(
        self, backrefs: Set[Key], refs: Set[Key], key: Key
    ) -> Dict[Any, Any]:
        """

        Compute local reference graph for a given item, and return a usefull
        representation to show a d3 graph in javascript

        backrefs:
            list of backreferences
        refs:
            list of forward references
        keys:
            current iten

        """
        weights = {}
        assert isinstance(backrefs, set)
        for b in backrefs:
            assert isinstance(b, Key)
        assert isinstance(refs, set)
        for f in refs:
            assert isinstance(f, Key)

        all_nodes = set(backrefs).union(set(refs))

        raw_edges = []
        for k in set(all_nodes):
            name = k.path
            neighbors_refs = self.store.get_backref(k)
            weights[name] = len(neighbors_refs)
            orig = [x.path for x in neighbors_refs]
            all_nodes = all_nodes.union(neighbors_refs)
            for o in orig:
                raw_edges.append((k.path, o))

        data: Dict[str, List[Any]] = {"nodes": [], "links": []}

        if len(weights) > 50:
            for thresh in sorted(set(weights.values())):
                log.debug("%s items ; remove items %s or lower", len(weights), thresh)
                weights = {k: v for k, v in weights.items() if v > thresh}
                log.debug("down to %s items", len(weights))
                if len(weights) < 50:
                    break

        all_nodes = set(all_nodes)
        nums_ = set()
        edges = list(raw_edges)
        nodes = list(set(weights.keys()))
        for a, b in edges:
            if (a not in nodes) or (b not in nodes):
                continue
            nums_.add(a)
            nums_.add(b)
        nums = {x: i for i, x in enumerate(nodes, start=1)}

        for i, (from_, to) in enumerate(edges):
            if from_ == to:
                continue
            if from_ not in nodes:
                continue
            if to not in nodes:
                continue
            if key[3] in (to, from_):
                continue
            data["links"].append({"source": nums[from_], "target": nums[to], "id": i})

        for node in nodes:
            diam = 8.0
            if node == key[3]:
                continue
                diam = 18.0
            elif node in weights:
                diam = 8.0 + math.sqrt(weights[node])

            candidates = [n for n in all_nodes if n[3] == node and "??" not in n]
            if not candidates:
                uu = None
            else:
                # TODO : be smarter when we have multiple versions. Here we try to pick the latest one.
                latest_version: Key = list(sorted(candidates))[-1]
                uu = self.resolver.resolve(RefInfo.from_untrusted(*latest_version))

            data["nodes"].append(
                {
                    "id": nums[node],
                    "val": diam,
                    "label": node,
                    "mod": ".".join(node.split(".")[0:1]),
                    "url": uu,
                }
            )
        return data

    async def index(self):
        keys = self.store.glob((None, None, "meta", "aliases.cbor"))
        libraries = {}
        from packaging.version import parse

        for k in keys:
            meta = encoder.decode(self.store.get_meta(k))
            if k.module in libraries:
                libraries[k.module][1].append(k.version)
            else:
                libraries[k.module] = (k.module, [k.version], meta["logo"])

        def p(v):
            if v == "??":
                return parse("0.0.0")
            else:
                return parse(v)

        data = [
            (a, list(sorted(b, reverse=True, key=p)), c)
            for (a, b, c) in libraries.values()
        ]

        return self.env.get_template("index.tpl.j2").render(data=data)

    async def _write_index(self, html_dir):
        if html_dir:
            (html_dir / "index.html").write_text(await self.index())

    async def virtual(self, module: str, node_name: str):
        """
        Endpoint that  look for all nodes of a specific type in all
        the known pages and render them.
        """
        _module: Optional[str]
        if module == "*":
            _module = None
        else:
            _module = module
        items = list(self.store.glob((_module, None, None, None)))

        visitor = TreeVisitor([getattr(take2, node_name)])
        acc = []
        for it in items:
            if it.kind in ("assets", "examples", "meta"):
                continue
            data = self.store.get(it)
            try:
                obj = encoder.decode(data)
            except Exception:
                print("Decode exception", it)
                continue
            if not isinstance(obj, IngestedBlobs):
                print("SKIP", it)
                continue
            lacc = []
            for a in obj.arbitrary + list(obj.content.values()):
                res = visitor.generic_visit(a)
                for v in res.values():
                    lacc.extend(v)
            if lacc:
                acc.append(Section(lacc, title=it[3]))
            # a2 = visitor.generic_visit(obj.content)
            # print(a1, a2)

        doc = IngestedBlobs.new()

        class S:
            value = None

        doc.signature = S()
        doc.arbitrary = acc
        return self.env.get_template("html.tpl.j2").render(
            graph=None,
            # TODO: next 2
            backrefs=[],
            module="*",
            doc=doc,
            parts=list({"*": []}.items()),
            version="*",
            ext="",
            current_type="",
            logo=None,
            meta={},
        )

    async def _write_gallery(self, config):
        """ """
        mv2 = self.store.glob((None, None))
        for _, (module, version) in self.progress(
            set(mv2), description="Rendering galleries..."
        ):
            # version, module = item.path.name, item.path.parent.name
            data = await self.gallery(
                module,
                version,
                ext=".html",
            )
            if config.output_dir:
                (config.output_dir / module / version / "gallery").mkdir(
                    parents=True, exist_ok=True
                )
                (
                    config.output_dir / module / version / "gallery" / "index.html"
                ).write_text(data)

    async def gallery(self, package: str, version: str, ext=""):
        # TODO FIX
        _package: Optional[str]
        _version: Optional[str]
        if ":" in package:
            _package, _ = package.split(":")
        else:
            _package = package
        _version = version
        if _package == "*" and _version == "*":
            _package = None
            _version = None

        figmap = defaultdict(lambda: [])
        assert isinstance(self.store, GraphStore)
        if _package is not None:
            meta = encoder.decode(
                self.store.get_meta(Key(_package, _version, None, None))
            )
        else:
            meta = {"logo": None}
        logo = meta["logo"]
        res = self.store.glob((_package, _version, "assets", None))
        backrefs: set[tuple[str, str, str, str]] = set()
        for key in res:
            brs = {tuple(x) for x in self.store.get_backref(key)}
            backrefs = backrefs.union(brs)

        for key2 in backrefs:
            data = encoder.decode(self.store.get(Key(*key2)))
            if "examples" in key2:
                continue
            # TODO: examples can actuallly be just Sections.
            assert isinstance(data, IngestedBlobs)
            i = data

            for k in [
                u.value for u in i.example_section_data if u.__class__.__name__ == "Fig"
            ]:
                package, v, kind, _path = key2
                # package, filename, link
                impath = f"{self.prefix}{k.module}/{k.version}/img/{k.path}"
                link = f"{self.prefix}/{package}/{v}/api/{_path}"
                # figmap.append((impath, link, name)
                figmap[package].append((impath, link, _path))

        glist = self.store.glob((package, _version, "examples", None))
        for target_key in glist:
            section = encoder.decode(self.store.get(target_key))

            for k in [
                u.value for u in section.children if u.__class__.__name__ == "Fig"
            ]:
                package, v, _, _path = target_key

                # package, filename, link
                impath = f"{self.prefix}{k.module}/{k.version}/img/{k.path}"
                link = f"{self.prefix}{package}/{v}/examples/{_path}"
                name = _path
                figmap[package].append((impath, link, name))

        class D:
            pass

        doc = D()
        pap_keys = self.store.glob((None, None, "meta", "papyri.cbor"))
        parts: Any = {package: []}
        for pk in pap_keys:
            mod, ver, kind, identifier = pk
            parts[package].append((RefInfo(mod, ver, "api", mod), mod))

        return self.env.get_template("gallery.tpl.j2").render(
            logo=logo,
            meta=meta,
            figmap=figmap,
            module=package,
            parts_mods=parts.get(package, []),
            # TODO: here
            parts=list(parts.items()),
            version=_version,
            parts_links=defaultdict(lambda: ""),
            doc=doc,
        )

    async def _get_toc_for(self, package, version):
        keys = self.store.glob((package, version, "meta", "toc.cbor"))
        assert len(keys) == 1, (keys, package, version)
        data = self.store.get(keys[0])
        return encoder.decode(data)

    async def _list_narative(self, package: str, version: str, ext=""):
        toctrees = await self._get_toc_for(package, version)

        meta = encoder.decode(self.store.get_meta(Key(package, version, None, None)))
        logo = meta["logo"]

        class D:
            pass

        doc = D()
        return self.env.get_template("toctree.tpl.j2").render(
            logo=logo,
            meta=meta,
            module=package,
            parts=list({}.items()),
            parts_mods=[],
            version=version,
            parts_links=defaultdict(lambda: ""),
            doc=doc,
            toctrees=toctrees,
        )

    def _myst_root(self, doc: IngestedBlobs) -> MRoot:
        """
        Convert a internal IngestedBlob document into a MyST tree
        for rendering.
        """
        to_suppress = []
        myst_acc: List[Any] = []
        if doc.signature:
            myst_acc.append(doc.signature)
            del doc.signature
        for k, v in doc.content.items():
            assert isinstance(v, Section)
            ct: List[Any]
            if v.children:
                if k in ("Extended Summary", "Summary"):
                    ct = []
                else:
                    ct = [MHeading(depth=1, children=[MText(k)])]
                ct.extend(v.children)
                myst_acc.extend(ct)
            else:
                to_suppress.append(k)
        for k in to_suppress:
            del doc.content[k]

        doc.arbitrary = [self.LR.visit(x) for x in doc.arbitrary]
        for a in doc.arbitrary:
            myst_acc.extend(a.children)
        if doc.see_also:
            myst_acc.extend(
                [
                    MHeading(
                        children=[MText("See Also")],
                        depth=1,
                    )
                ]
                + [DefList([self.LR.visit(s) for s in doc.see_also])]
            )
        if doc.example_section_data:
            ct = [MHeading(depth=1, children=[MText("Examples")])]
            ct.extend([self.LR.visit(x) for x in doc.example_section_data])
            ct = [MParagraph([x]) if isinstance(x, MText) else x for x in ct]
            myst_acc.extend(ct)

        del doc.arbitrary
        del doc.see_also
        return MRoot(myst_acc)

    def render_one(
        self,
        template: Template,
        doc: IngestedBlobs,
        qa: str,
        *,
        current_type: str,
        backrefs: List[RefInfo],
        parts: Dict[str, List[Tuple[str, str]]] = dict(),
        parts_links=(),
        graph: str = "{}",
        meta: dict,
        toctrees,
    ):
        """
        Return the rendering of one document

        Parameters
        ----------
        template
            a Jinja template object used to render.
        doc : DocBlob
            a Doc object with the informations for current obj
        qa : str
            fully qualified name for current object
        backrefs : list of str
            backreferences of document pointing to this.
        parts : Dict[str, list[Tuple[str, str]]
            used for navigation and for parts of the breakcrumbs to have navigation to siblings.
            This is not directly related to current object.
        parts_links : <Insert Type here>
            <Multiline Description Here>
        graph : <Insert Type here>
            <Multiline Description Here>
        logo : <Insert Type here>
            <Multiline Description Here>

        """
        assert template is not None

        assert isinstance(meta, dict)
        # TODO : move this to ingest likely.
        # Here if we have too many references we group them on where they come from.
        assert not hasattr(doc, "logo")

        mback = backrefs_to_myst(backrefs, self.LR)

        root = self._myst_root(doc)
        root.children.extend(mback)

        root_json = json.dumps(root.to_dict(), indent=2)
        try:
            module = qa.split(".")[0]
            return template.render(
                current_type=current_type,
                myst_root=root_json,
                item_line=doc.item_line,
                item_file=doc.item_file,
                logo=meta.get("logo", None),
                version=meta["version"],
                module=module,
                name=qa.split(":")[-1].split(".")[-1],
                # TODO: next 1
                parts_mods=parts.get(module, []),
                parts=list(parts.items()),
                parts_links=parts_links,
                graph=graph,
                meta=meta,
                toctrees=toctrees,
            )
        except Exception as e:
            e.add_note(f"Rendering with QA={qa}")
            raise

    async def _serve_narrative(self, package: str, version: str, ref: str):
        """
        Serve the narrative part of the documentation for given package
        """
        # return "Not Implemented"
        key = Key(package, version, "docs", ref)
        bytes = self.store.get(key)
        doc_blob = encoder.decode(bytes)
        meta = encoder.decode(self.store.get_meta(key))
        # return "OK"

        template = self.env.get_template("html.tpl.j2")

        assert isinstance(doc_blob, IngestedBlobs), type(doc_blob)

        toctrees = await self._get_toc_for(package, version)

        def open_toctree(toc, path):
            """
            Temporary, we really need a custom object.
            This mark specific toctree elements to be open in the rendering.
            Typically all nodes that link to current page.
            """
            for c in toc.children:
                if open_toctree(c, path):
                    toc.open = True
                # else:
                #    toc.open = False
            if toc.ref.path == path:
                toc.open = False
                toc.current = True
                return True
            return toc.open

        for t in toctrees:
            open_toctree(t, ref)

        return self.render_one(
            current_type="docs",
            meta=meta,
            template=template,
            doc=doc_blob,
            qa=package,  # incorrect
            parts={package: [], ref: []},
            parts_links={},
            backrefs=[],
            graph="{}",
            toctrees=toctrees,
        )

    async def _route_data(self, ref, version, known_refs):
        ref = ref.split("/")[0]
        if ":" in ref:
            modroot, _ = ref.split(":")
        else:
            modroot = ref

        root = modroot.split(".")[0]
        key = Key(root, version, "module", ref)
        gbytes, backward, forward = self.store.get_all(key)
        x_, y_ = find_all_refs(self.store)
        doc_blob = encoder.decode(gbytes)
        return x_, y_, doc_blob, backward, forward

    async def _route(
        self,
        ref: str,
        version: Optional[str] = None,
    ):
        assert not ref.endswith(".html")
        assert version is not None
        assert ref != ""

        template = self.env.get_template("html.tpl.j2")
        if ":" in ref:
            modroot, _ = ref.split(":")
        else:
            modroot = ref
        root = modroot.split(".")[0]
        meta = encoder.decode(self.store.get_meta(Key(root, version, None, None)))

        known_refs, ref_map = find_all_refs(self.store)

        # technically incorrect we don't load backrefs
        x_, y_, doc_blob, backward, forward = await self._route_data(
            ref, version, known_refs
        )
        assert x_ == known_refs
        assert y_ == ref_map
        assert version is not None

        siblings = compute_siblings_II(ref, known_refs)

        url_sib: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        for k, v in siblings.items():
            url_sib[k] = [(self.resolver.must_resolve(ref), name) for ref, name in v]

        # End computing siblings.
        if True:  # handle if thing don't exists.
            # The reference we are trying to view exists;
            # we will now just render it.
            assert root is not None
            # assert version is not None
            data = self.compute_graph(
                backward, forward, Key(root, version, "module", ref)
            )
            json_str = json.dumps(data)
            parts_links = {}
            acc = ""
            for k in siblings.keys():
                acc += k
                parts_links[k] = acc
                acc += "."
            backrefs = [RefInfo(*k) for k in backward]
            return self.render_one(
                current_type="api",
                template=template,
                doc=doc_blob,
                qa=ref,
                parts=url_sib,
                parts_links=parts_links,
                backrefs=backrefs,
                graph=json_str,
                meta=meta,
                toctrees=[],
            )
        else:
            # The reference we are trying to render does not exists
            # TODO
            error = self.env.get_template("404.tpl.j2")
            return error.render(backrefs=list(set()), tree={}, ref=ref, module=root)

    async def _write_api_file(
        self,
        tree: Any,
        known_refs: Any,
        ref_map: Any,
        config: Any,
        graph: Any,
    ):
        template = self.env.get_template("html.tpl.j2")
        gfiles = list(self.store.glob((None, None, "module", None)))
        random.shuffle(gfiles)
        for _, key in progress(gfiles, description="Rendering API..."):
            module, version = key.module, key.version
            if config.ascii:
                _rich_ascii(key, store=self.store)
            if config.html:
                doc_blob, qa, siblings, parts_links, backward, forward = await loc(
                    key,
                    store=self.store,
                    tree=tree,
                    known_refs=known_refs,
                    ref_map=ref_map,
                )
                url_sib: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
                for k, v in siblings.items():
                    url_sib[k] = [
                        (self.resolver.must_resolve(ref), name) for ref, name in v
                    ]
                backward_r = [RefInfo(*x) for x in backward]
                if graph:
                    data = self.compute_graph(set(backward), set(forward), key)
                else:
                    data = {}
                json_str = json.dumps(data)
                meta = encoder.decode(self.store.get_meta(key))
                html: str = self.render_one(
                    current_type="API",
                    template=template,
                    doc=doc_blob,
                    qa=qa,
                    parts=url_sib,
                    parts_links=parts_links,
                    backrefs=backward_r,
                    graph=json_str,
                    meta=meta,
                    toctrees=[],
                )
                if config.output_dir:
                    (config.output_dir / module / version / "api").mkdir(
                        parents=True, exist_ok=True
                    )
                    tfile = config.output_dir / module / version / "api" / f"{qa}.html"
                    if config.minify:
                        tfile.write_text(minify(html))
                    else:
                        tfile.write_text(html)

    async def _copy_dir(self, src_dir: Path, dest_dir: Path):
        assert dest_dir.exists()
        for item in src_dir.glob("*"):
            dest_item = dest_dir / item.name
            if item.is_file():
                bts = item.read_bytes()
                dest_item.write_bytes(bts)
            else:
                dest_item.mkdir(exist_ok=True)
                await self._copy_dir(item, dest_item)

    async def copy_static(self, output_dir):
        here = Path(os.path.dirname(__file__))
        _static = here / "static"
        if output_dir is not None:
            output_dir.mkdir(exist_ok=True)
            static = output_dir.parent / "static"
            static.mkdir(exist_ok=True)
            await self._copy_dir(_static, static)
            (static / "pygments.css").write_bytes(await pygment_css().get_data())

    async def copy_assets(self, config):
        """
        Copy assets from to their final destination.

        Assets are all the binary files that we don't want to change.
        """
        if config.output_dir is None:
            return

        assets_2 = self.store.glob((None, None, "assets", None))
        for _, asset in dummy_progress(assets_2, description="Copying assets"):
            b = config.output_dir / asset.module / asset.version / "img"
            b.mkdir(parents=True, exist_ok=True)
            data = self.store.get(asset)
            (b / asset.path).write_bytes(data)

    async def _write_example_files(self, config):
        if not config.html:
            return

        examples = list(self.store.glob((None, None, "examples", None)))
        for _, example in progress(examples, description="Rendering Examples..."):
            module, version, _, path = example
            data = await self.render_single_examples(
                module,
                version,
                data=self.store.get(example),
            )
            if config.output_dir:
                (config.output_dir / module / version / "examples").mkdir(
                    parents=True, exist_ok=True
                )
                (
                    config.output_dir / module / version / "examples" / f"{path}.html"
                ).write_text(data)

    async def _write_narrative_files(self, config):
        narrative = list(self.store.glob((None, None, "docs", None)))
        for _, (module, version, _, path) in progress(
            narrative, description="Rendering Narrative..."
        ):
            try:
                data = await self._serve_narrative(module, version, path)
            except Exception as e:
                e.add_note(
                    f"rendering {module=} {version=} {path=}",
                )
                raise
            if config.output_dir:
                (config.output_dir / module / version / "docs").mkdir(
                    parents=True, exist_ok=True
                )
                (
                    config.output_dir / module / version / "docs" / f"{path}.html"
                ).write_text(data)

        tocs = set([(m, v) for m, v, _, _ in narrative])
        for module, version in tocs:
            print("toc for", module, version)
            toc = await self._list_narative(module, version, "")
            if config.output_dir:
                (config.output_dir / module / version / "docs" / "toc.html").write_text(
                    toc
                )

    async def examples_handler(self, package, version, subpath):
        meta = encoder.decode(self.store.get_meta(Key(package, version, None, None)))

        pap_keys = self.store.glob((None, None, "meta", "aliases.cbor"))
        parts = {package: []}
        for pk in pap_keys:
            mod, ver, _, _ = pk
            parts[package].append((RefInfo(mod, ver, "api", mod), mod))

        bytes_ = self.store.get(Key(package, version, "examples", subpath))

        ex = encoder.decode(bytes_)
        assert isinstance(ex, Section)

        class Doc:
            pass

        doc = Doc()

        return self.env.get_template("examples.tpl.j2").render(
            meta=meta,
            logo=meta["logo"],
            module=package,
            parts=list(parts.items()),
            version=version,
            parts_links=defaultdict(lambda: ""),
            doc=doc,
            ex=ex,
        )

    async def render_single_examples(self, module, version, *, data):
        mod_vers = self.store.glob((None, None))
        meta = encoder.decode(self.store.get_meta(Key(module, version, None, None)))
        logo = meta["logo"]
        parts = {module: []}
        for mod, ver in mod_vers:
            assert isinstance(mod, str)
            assert isinstance(ver, str)
            parts[module].append((RefInfo(mod, ver, "api", mod), mod))

        ex = encoder.decode(data)

        class Doc:
            pass

        doc = Doc()

        return self.env.get_template("examples.tpl.j2").render(
            meta=meta,
            logo=logo,
            module=module,
            # TODO: here
            parts_mods=parts.get(module, []),
            parts=list(parts.items()),
            version=version,
            parts_links=defaultdict(lambda: ""),
            doc=doc,
            ex=ex,
        )


async def img(package, version, subpath=None) -> Response:
    folder = ingest_dir / package / version / "assets"
    return await send_from_directory(folder, subpath)


def static(name) -> Callable[[], bytes]:
    here = Path(os.path.dirname(__file__))
    static = here / "static"

    async def f():
        return await send_from_directory(static, name)

    f.__name__ = name

    return f


def pygment_css() -> Response:
    return Response(CSS_DATA, mimetype="text/css")


async def serve_app(subpath):
    print("subpath...", subpath)
    import glob

    here = Path(os.path.dirname(__file__))
    if "main" in subpath:
        ext = subpath.split(".")[-1]

        res = glob.glob(glob.glob(f"{here}/app/static/{ext}/main.*.{ext}")[0])
        new = "/".join(res[0].split("/")[-2:])
        print("Did you mean ", subpath, new)
        subpath = new

    static = here / "app" / "static"
    return await send_from_directory(static, subpath)


def serve(*, sidebar: bool, port=1234):
    app = QuartTrio(__name__, static_folder=None)

    gstore = GraphStore(ingest_dir)
    prefix = "/p/"
    html_renderer = HtmlRenderer(
        gstore, sidebar=sidebar, prefix=prefix, trailing_html=False
    )

    async def full(package, version, ref):
        if version == "*":
            res = list(html_renderer.store.glob([package, None]))
            assert len(res) == 1
            version = res[0][1]
            return redirect(f"{prefix}{package}/{version}/api/{ref}")
            # print(list(html_renderer.store.glob(
        return await html_renderer._route(ref, version)

    async def g(module):
        return await html_renderer.gallery(module)

    async def gr():
        return await html_renderer.gallery("*", "*")

    app.route("/logo.png")(static("papyri-logo.png"))
    app.route("/static/pygments.css")(pygment_css)
    app.route("/app/<path:subpath>")(serve_app)
    # sub here is likely incorrect
    app.route(f"{prefix}<package>/<version>/img/<path:subpath>")(img)
    app.route(f"{prefix}<package>/<version>/examples/<path:subpath>")(
        html_renderer.examples_handler
    )
    app.route(f"{prefix}<package>/<version>/gallery")(html_renderer.gallery)
    app.route(f"{prefix}<package>/<version>/toc/")(html_renderer._list_narative)
    app.route(f"{prefix}<package>/<version>/docs/")(
        lambda package, version: redirect(f"{prefix}{package}/{version}/docs/index")
    )
    app.route(f"{prefix}<package>/<version>/docs/<ref>")(html_renderer._serve_narrative)
    app.route(f"{prefix}<package>/<version>/api/<ref>")(full)
    app.route(f"{prefix}<package>/static/<path:subpath>")(full)
    app.route(f"{prefix}/gallery/")(gr)
    app.route(f"{prefix}/gallery/<module>")(g)
    app.route(f"{prefix}/virtual/<module>/<node>")(html_renderer.virtual)
    app.route("/")(html_renderer.index)

    async def serve_static(path):
        here = Path(os.path.dirname(__file__))
        static = here / "static"
        return await send_from_directory(static, path)

    app.route("/static/<path:path>")(serve_static)

    port = int(os.environ.get("PORT", port))
    print("Seen config port ", port)
    prod = os.environ.get("PROD", None)
    if prod:
        app.run(port=port, host="0.0.0.0")
    else:
        app.run(port=port)


class Resolver:
    # mapping from package name to existing (version(s))
    # currently multiple version nor really supported.
    # this is used when we request to reach to a page from any version of a
    # library to know which one we should look into.
    version: Dict[str, str]

    # the rendering might under prefix, depending on how it is hosted.
    # This should thus be prepended to generated urls.
    # it should end and start with a `/`.
    prefix: str

    # Extension that need to be added to the end of the url, why many server
    # are fine omitting `.html`, it might be necessary when resolving for
    # statically generated contents.

    extension: str

    _cache: Dict[RefInfo, Tuple[bool, Optional[str]]] = {}

    def __init__(self, store, prefix: str, extension: str) -> None:
        """
        Given a RefInfo to an object, resolve it to a full http-link
        with the current configuration.

        Parameters
        ----------
        store:
            current store which knows about the current existing objects, and
            references.
        prefix:
            url prefix that should be prepended when the documentation is hosted
            at a subpath.
        extension:
            Depending on the context, (ssg, or webserver), links may need an
            explicit extension.

        """
        if extension != "":
            assert extension.startswith(".")
        assert prefix.startswith("/")
        assert prefix.endswith("/")
        self.store = store
        self.prefix = prefix
        self.extension = extension

        self.version = {}

        for p, v in {
            (package, version)
            for (package, version, _, _) in self.store.glob((None, "*", "meta", None))
        }:
            if p in self.version:
                pass
                # TODO:, likely parse version here if possible.
                # maxv = max(v, self.version[p])
                # print("multiple version for package", p, "Trying most recent", maxv)
            self.version[p] = v

    def exists_resolve(self, info: RefInfo) -> Tuple[bool, Optional[str]]:
        """Resolve a RefInfo to a URL, additionally return wether the link item exists

        Return
        ------
        exists: boolean
            Whether  the target document exists
        url : str|None
            If exists, URL where target document can be found

        """

        if info in self._cache:
            return self._cache[info]

        module, version_number, kind, path = info
        if kind == "api":
            kind = "module"
        if version_number == "*":
            if info.module in self.version:
                version_number = self.version[info.module]

        query_ref = RefInfo(module, version_number, kind, path)
        # TODO: Fix, for example in IPython narrative docs the
        # toctree point to ``pr/*`` which is a subfolder we don't support yet:
        #
        # .. toctree::
        #     :maxdepth: 2
        #     :glob:
        #
        #     pr/*
        if kind == "?":
            self._cache[info] = (False, None)
            return False, None
        sgi = self.store.glob(query_ref)
        if sgi:
            assert len(sgi) == 1, (
                "we have no reason to have more than one reference",
                sgi,
            )
            exists, url = self._resolve(query_ref)
            self._cache[info] = (exists, url)
            return exists, url

        # we may want to find older versions.

        self._cache[info] = (False, None)
        return False, None

    def resolve(self, info: RefInfo) -> Optional[str]:
        exists, url = self.exists_resolve(info)
        # TODO: this is moslty used to render navigation we should make sure that
        # links are resolved and exists before rendering.
        # assert exists, f"{info=} doe not exists"
        return url

    def must_resolve(self, info: RefInfo) -> str:
        exists, url = self.exists_resolve(info)
        if not exists:
            return "??"
        assert exists, info
        assert url is not None
        return url

    def _resolve(self, info) -> Tuple[bool, str]:
        """
        TODO : refactor this in a better  way so that the link reifier can know whether the link will resolve.
        """
        assert isinstance(info, RefInfo), info
        assert info.kind in (
            "module",
            "api",
            "examples",
            "assets",
            # "?",
            "docs",
            # TODO:
            "to-resolve",
        ), repr(info)
        # assume same package/version for now.
        # assert info.version is not "*", info
        assert info.module is not None
        version_number = info.version
        if version_number == "*":
            if info.module in self.version:
                version_number = self.version[info.module]
        if info.module is None:
            assert info.version is None
            return True, info.path + self.extension
        if info.kind == "module":
            return (
                True,
                f"{self.prefix}{info.module}/{version_number}/api/{info.path}{self.extension}",
            )
        if info.kind == "examples":
            return (
                True,
                f"{self.prefix}{info.module}/{version_number}/examples/{info.path}{self.extension}",
            )
        else:
            return (
                True,
                f"{self.prefix}{info.module}/{version_number}/{info.kind}/{info.path}{self.extension}",
            )


class LinkReifier(TreeReplacer):
    def __init__(self, resolver):
        self.resolver = resolver
        super().__init__()

    def replace_Link(self, link: Link):
        """
        By default our links resolution is delayed,
        Here we resolve them.

        Some of this resolution should be moved to earlier.
        """
        if link.reference.kind == "local":
            return [
                MLink(
                    # TODO: what do we do for reference in same document
                    children=[MText(link.value)],
                    url=f"#{link.reference.path}",
                    title=str(link.reference),
                )
            ]
        else:
            exists, turl = self.resolver.exists_resolve(link.reference)
            if exists:
                return [
                    MLink(
                        children=[MText(link.value)],
                        url=turl,
                        title="",
                    )
                ]
            else:
                return [
                    MUnimpl(
                        [
                            # TODO new non-existing text category ?
                            MText(link.value)
                        ]
                    )
                ]

    def replace_Fig(self, data: Fig):
        # exists, url = self.resolver.exists_resolve(fig.value)
        url = f"/p/{data.value.module}/{data.value.version}/img/{data.value.path}"
        return [MImage(url, alt="")]

    def replace_Section(self, section: Section) -> List[MRoot]:
        ch = [self.visit(c) for c in section.children]
        if section.title is not None and section.title.lower() not in (
            "extended summary",
            "summary",
        ):
            h = [MHeading(depth=section.level, children=[MText(section.title)])]
        else:
            h = []
        return [MRoot(h + ch)]

    def replace_SeeAlsoItem(self, see_also: SeeAlsoItem) -> List[DefListItem]:
        name = see_also.name
        descriptions = see_also.descriptions

        name = self.visit(name)
        descriptions = [self.visit(d) for d in descriptions]

        # TODO: this is type incorrect for now. Fix later
        return [DefListItem(dt=name, dd=descriptions)]

    def replace_RefInfo(self, refinfo: RefInfo) -> List[Any]:
        """
        By default our links resolution is delayed,
        Here we resolve them.

        Some of this resolution should be moved to earlier.
        """
        exists, turl = self.resolver.exists_resolve(refinfo)
        if exists:
            return [MLink(children=[MText(refinfo.path)], url=turl, title=refinfo.path)]
        else:
            return [MText(refinfo.path + "(?)")]


def _rich_render(key: Key, store: GraphStore) -> List[Any]:
    from .rich_render import RichVisitor

    doc = encoder.decode(store.get(key))
    resolver = Resolver(store, prefix="/p/", extension="")
    LR = LinkReifier(resolver=resolver)
    RV = RichVisitor()
    to_del = []

    for k, v in doc.content.items():
        if v.children:
            ct = MRoot([MHeading(depth=1, children=[MText(k)]), *v.children])
            doc.content[k] = RV.visit(LR.visit(ct))
        else:
            to_del.append(k)
    for k in to_del:
        del doc.content[k]
    name = key.path.split(":")[-1].split(".")[-1]
    if doc.signature:
        myst_acc = [name + str(doc.signature.to_signature())]
    else:
        myst_acc = []

    myst_acc += [RV.visit(LR.visit(x)) for x in doc.arbitrary]
    myst_acc += [doc.content[k] for k in doc.content]

    if doc.see_also:
        myst_acc.append(
            RV.visit(
                MRoot(
                    [
                        MHeading(
                            children=[MText("See Also")],
                            depth=1,
                        )
                    ]
                    + [DefList([LR.visit(s) for s in doc.see_also])]
                )
            )
        )
    return myst_acc


from rich.theme import Theme

THEME = Theme(
    {
        "m.inline_code": "bold blue",
        "unimp": "red",
        "m.directive": "cyan",
        "param": "green",
        "param_type": "yellow",
        "math": "green",
    }
)


async def rich_render(
    name: str, width: Optional[int] = None, color: bool = True
) -> None:
    store = GraphStore(ingest_dir, {})
    key = next(iter(store.glob((None, None, "module", name))))

    from rich.console import Console

    console = Console(
        theme=THEME,
        width=width,
        no_color=not color,
        color_system="256" if color else None,
    )
    try:
        for it in _rich_render(key, store):
            console.print(it)
    except Exception as e:
        e.add_note(f"rendering {key=}")
        raise


def _rich_ascii(key: Key, store=None):
    if store is None:
        store = GraphStore(ingest_dir, {})

    from rich.console import Console

    from io import StringIO

    buf = StringIO()

    console = Console(
        theme=THEME,
        record=True,
        file=buf,
        no_color=True,
        width=80,
    )

    try:
        for it in _rich_render(key, store):
            console.print(it)
    except Exception as e:
        e.add_note(f"rendering {key=}")
        raise

    lines = console.export_text().splitlines()
    return "\n".join(l.rstrip() for l in lines)


async def loc(document: Key, *, store: GraphStore, tree, known_refs, ref_map):
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
    try:
        assert isinstance(document, Key), type(document)
        qa = document.path
        bytes_, backward, forward = store.get_all(document)
        doc_blob: IngestedBlobs = encoder.decode(bytes_)
    except Exception as e:
        e.add_note(f"Reading {document.path}")
        raise

    siblings = cs2(qa, tree, ref_map)

    parts_links = {}
    acc = ""
    for k in siblings.keys():
        acc += k
        parts_links[k] = acc
        acc += "."
    try:
        return doc_blob, qa, siblings, parts_links, backward, forward
    except Exception as e:
        raise type(e)(f"Error in {qa}") from e


@dataclass
class StaticRenderingConfig:
    """Class for keeping track of an item in inventory."""

    html: bool
    html_sidebar: bool
    ascii: bool
    output_dir: Optional[Path]
    minify: bool


async def main(ascii: bool, html, dry_run, sidebar: bool, graph: bool, minify: bool):
    """
    This does static rendering of all the given files.

    Parameters
    ----------
    ascii: bool
        whether to render ascii files.
    html: bool
        whether to render the html
    dry_run: bool
        do not write the output.
    Sidebar:bool
        render the sidebar in html
    graph: bool

    """

    html_dir_: Optional[Path] = default_config.html_dir
    if dry_run:
        output_dir = None
        html_dir_ = None
    else:
        assert html_dir_ is not None
        output_dir = html_dir_ / "p"
        output_dir.mkdir(exist_ok=True)
    config = StaticRenderingConfig(html, sidebar, ascii, output_dir, minify)
    prefix = "/p/"

    gstore = GraphStore(ingest_dir, {})

    known_refs, ref_map = find_all_refs(gstore)
    # end

    family = frozenset(_.path for _ in known_refs)

    tree = make_tree(family)
    if html_dir_ is not None:
        log.info("going to erase %s", html_dir_)
        shutil.rmtree(html_dir_)
    else:
        log.info("no output dir, we'll try not to touch the filesystem")

    # shuffle files to detect bugs, just in case.
    # Gallery

    html_renderer = HtmlRenderer(
        gstore, sidebar=config.html_sidebar, prefix=prefix, trailing_html=True
    )

    await html_renderer._write_api_file(
        tree,
        known_refs,
        ref_map,
        config,
        graph,
    )
    await html_renderer._write_gallery(config)

    await html_renderer._write_example_files(config)
    await html_renderer._write_index(html_dir_)
    await html_renderer.copy_assets(config)
    await html_renderer.copy_static(config.output_dir)
    await html_renderer._write_narrative_files(config)
