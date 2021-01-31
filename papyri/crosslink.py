from __future__ import annotations

import builtins
import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from glob import escape as ge
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

from there import print

from .config import ingest_dir
from .gen import DocBlob, normalise_ref
from .stores import Store
from .take2 import Directive, Link, Node, Param, RefInfo, Section, SeeAlsoItem
from .utils import progress

warnings.simplefilter("ignore", UserWarning)


def find_all_refs(store):
    o_family = sorted(list(store.glob("*/*/module/*.json")))

    # TODO
    # here we can't computejust the dictionary and use frozenset(....values())
    # as we may have multiple version of libraries; this is something that will
    # need to be fixed in the long run
    known_refs = []
    ref_map = {}
    for item in o_family:
        module, v = item.path.parts[-4:-2]
        r = RefInfo(module, v, "api", item.name[:-5])
        known_refs.append(r)
        ref_map[r.path] = r
    return frozenset(known_refs), ref_map


@dataclass
class IngestedBlobs(Node):

    __slots__ = ("backrefs",) + (
        "_content",
        "refs",
        "ordered_sections",
        "item_file",
        "item_line",
        "item_type",
        "aliases",
        "example_section_data",
        "see_also",
        "version",
        "signature",
        "references",
        "logo",
    )

    _content: Dict[str, Section]
    refs: List[Link]
    ordered_sections: List[str]
    item_file: Optional[str]
    item_line: Optional[int]
    item_type: Optional[str]
    aliases: List[str]
    example_section_data: Section
    see_also: List[SeeAlsoItem]  # see also data
    version: str
    signature: Optional[str]
    references: Optional[List[str]]
    logo: Optional[str]
    backrefs: List[str]

    __isfrozen = False

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backrefs = []
        self._content = None
        self.example_section_data = None
        self.refs = None
        self.ordered_sections = None
        self.item_file = None
        self.item_line = None
        self.item_type = None
        self.aliases = []

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    @property
    def content(self):
        """
        List of sections in the doc blob docstrings

        """
        return self._content

    @content.setter
    def content(self, new):
        assert not new.keys() - {
            "Signature",
            "Summary",
            "Extended Summary",
            "Parameters",
            "Returns",
            "Yields",
            "Receives",
            "Raises",
            "Warns",
            "Other Parameters",
            "Attributes",
            "Methods",
            "See Also",
            "Notes",
            "Warnings",
            "References",
            "Examples",
            "index",
        }
        self._content = new

    def process(self, qa, known_refs, aliases, verbose=True):
        local_refs = []
        sections_ = [
            "Parameters",
            "Returns",
            "Raises",
            "Yields",
            "Attributes",
            "Other Parameters",
            "Warns",
            ##
            "Warnings",
            "Methods",
            # "Summary",
            "Receives",
            # "Notes",
            # "Signature",
            #'Extended Summary',
            #'References'
            #'See Also'
            #'Examples'
        ]
        for s in sections_:

            local_refs = local_refs + [
                [u.strip() for u in x[0].split(",")]
                for x in self.content[s]
                if isinstance(x, Param)
            ]

        def flat(l):
            return [y for x in l for y in x]

        local_refs = frozenset(flat(local_refs))

        assert isinstance(known_refs, frozenset)

        visitor = DirectiveVisiter(qa, known_refs, local_refs, aliases)
        for section in ["Extended Summary", "Summary", "Notes"] + sections_:
            assert section in self.content
            self.content[section] = visitor.visit(self.content[section])
        if (len(visitor.local) or len(visitor.total)) and verbose:
            # TODO: reenable assert len(visitor.local) == 0, f"{visitor.local} | {qa}"
            print(f"Newly found links: {len(visitor.total)}, {qa}, {visitor.total}")
        self.example_section_data = visitor.visit(self.example_section_data)

        for d in self.see_also:
            new_desc = []
            for dsc in d.descriptions:
                new_desc.append(visitor.visit(dsc))
            d.descriptions = new_desc

    @classmethod
    def from_json(cls, data):
        inst = super().from_json(data)
        inst._freeze()
        return inst


@lru_cache
def _into(
    known_refs: List[Union[RefInfo, str]]
) -> Tuple[FrozenSet[RefInfo], FrozenSet[str]]:
    assert isinstance(known_refs, frozenset)
    k_path_map = frozenset({k.path for k in known_refs})
    return k_path_map


@lru_cache
def root_start(root, refs):
    return frozenset(r for r in refs if r.startswith(root))


@lru_cache(10000)
def endswith(end, refs):
    return frozenset(r for r in refs if r.endswith(end))


def resolve_(
    qa: str, known_refs: FrozenSet[RefInfo], local_refs: FrozenSet[str], ref: str, rev_aliases=None
) -> RefInfo:
    # RefInfo(module, version, kind, path)
    bq = False
    hash(known_refs)
    hash(local_refs)
    if rev_aliases is None:
        rev_aliases = {}
    if ref in rev_aliases:
        new_ref = rev_aliases[ref]
        #print(f'now looking for {new_ref} instead of {ref}')
        assert new_ref not in rev_aliases, 'would loop....'
        # TODOlikely can drop rev_aliases here
        res = resolve_(qa, known_refs, local_refs, new_ref, rev_aliases)
        print('   ', res)
        return res

    assert isinstance(ref, str), ref
    k_path_map = _into(known_refs)

    if ref.startswith("builtins."):
        return RefInfo(None, None, "missing", ref)
    if ref.startswith("str."):
        return RefInfo(None, None, "missing", ref)
    if ref in {"None", "False", "True"}:
        return RefInfo(None, None, "missing", ref)
    # here is sphinx logic.
    # https://www.sphinx-doc.org/en/master/_modules/sphinx/domains/python.html?highlight=tilde
    # tilda ~ hide the module name/class name
    # dot . search more specific first.
    if ref.startswith("~"):
        ref = ref[1:]
    if ref in local_refs:
        return RefInfo(None, None, "local", ref)
    if ref in k_path_map:
        return RefInfo(None, None, "exists", ref)
    else:
        if ref.startswith("."):
            if (found := qa + ref) in k_path_map:
                return RefInfo(None, None, "exists", found)
            else:
                root = qa.split(".")[0]
                sub1 = root_start(root, k_path_map)
                subset = endswith(ref, sub1)
                if len(subset) == 1:
                    return RefInfo(None, None, "exists", next(iter(subset)))
                else:
                    if len(subset) > 1:
                        # ambiguous ref
                        # print("subset:", ref)
                        pass

                # print(f"did not resolve {qa} + {ref}")
                return RefInfo(None, None, "missing", ref)

        parts = qa.split(".")
        for i in range(len(parts)):
            attempt = ".".join(parts[:i]) + "." + ref
            if attempt in k_path_map:
                return RefInfo(None, None, "exists", attempt)

    q0 = qa.split(".")[0]
    rs = root_start(q0, k_path_map)
    attempts = [q for q in rs if (ref in q)]
    if len(attempts) == 1:
        return RefInfo(None, None, "exists", attempts[0])
    return RefInfo(None, None, "missing", ref)


def load_one_uningested(
    bytes_: bytes, bytes2_: Optional[bytes], qa, known_refs, aliases
) -> IngestedBlobs:
    """
    Load the json from a DocBlob and make it an ingested blob.
    """
    data = json.loads(bytes_)

    old_data = DocBlob.from_json(data)

    blob = IngestedBlobs()
    # TODO: here or maybe somewhere else:
    # see also 3rd item description is improperly deserialised as now it can be a paragraph.
    # Make see Also an auto deserialised object in take2.
    blob.see_also = old_data.see_also

    for k in old_data.slots():
        setattr(blob, k, getattr(old_data, k))

    blob.refs = data.pop("refs", [])
    if bytes2_ is not None:
        backrefs = json.loads(bytes2_)
    else:
        backrefs = []
    blob.backrefs = backrefs
    blob.version = data.pop("version", "")

    blob.see_also = list(sorted(set(blob.see_also), key=lambda x: x.name.name))
    blob.example_section_data = blob.example_section_data
    blob.refs = list(sorted(set(blob.refs)))

    sections_ = [
        "Parameters",
        "Returns",
        "Raises",
        "Yields",
        "Attributes",
        "Other Parameters",
        "Warns",
        ##"Warnings",
        "Methods",
        # "Summary",
        "Receives",
    ]

    _local_refs: List[List[str]] = []

    for s in sections_:

        _local_refs = _local_refs + [
            [u.strip() for u in x[0].split(",")]
            for x in blob.content[s]
            if isinstance(x, Param)
        ]

    def flat(l) -> List[str]:
        return [y for x in l for y in x]

    local_refs: FrozenSet[str] = frozenset(flat(_local_refs))

    visitor = DirectiveVisiter(qa, frozenset(), local_refs, aliases=aliases)
    for section in ["Extended Summary", "Summary", "Notes"] + sections_:
        assert section in blob.content
        blob.content[section] = visitor.visit(blob.content[section])

    blob.process(qa, known_refs=known_refs, aliases=aliases, verbose=False)

    if blob.refs:
        new_refs: List[Link] = []
        kind = "exists"
        for value in blob.refs:
            assert isinstance(value, str)
            r = resolve_(qa, known_refs, frozenset(), value)
            new_refs.append(Link(value, r, kind, r.kind != "missing"))

        blob.refs = new_refs
    return blob


class TreeReplacer:
    def visit(self, node):
        assert not isinstance(node, list)
        res = self.generic_visit(node)
        assert len(res) == 1
        return res[0]

    def generic_visit(self, node) -> List[Node]:
        assert node is not None
        try:
            name = node.__class__.__name__
            if method := getattr(self, "replace_" + name, None):
                new_nodes = method(node)
            elif name in [
                "Word",
                "Verbatim",
                "Example",
                "BlockVerbatim",
                "Math",
                "Link",
                "Code",
                "Fig",
                "Words",
            ]:
                return [node]
            elif name in ["Text"]:
                assert False, "Text still present"
            else:
                new_children = []
                for c in node.children:
                    assert c is not None, f"{node=} has a None child"
                    replacement = self.generic_visit(c)
                    assert isinstance(replacement, list)
                    new_children.extend(replacement)
                node.children = new_children
                new_nodes = [node]
            assert isinstance(new_nodes, list)
            return new_nodes
        except Exception as e:
            raise type(e)(f"{node=}")


class DirectiveVisiter(TreeReplacer):
    def __init__(self, qa, known_refs: FrozenSet[RefInfo], local_refs, aliases):
        assert isinstance(qa, str), qa
        assert isinstance(known_refs, (list, set, frozenset)), known_refs
        self.known_refs = frozenset(known_refs)
        self.local_refs = frozenset(local_refs)
        self.qa = qa
        self.local: List[str] = []
        self.total: List[Tuple[Any, str]] = []
        # long -> short
        self.aliases: Dict[str, str] = aliases
        # short -> long
        self.rev_aliases = {v:k for k,v in aliases.items()}

    def replace_BlockDirective(self, block_directive):
        name = block_directive.directive_name
        if name not in {
            "math",
            "note",
            "versionadded",
            "versionchanged",
            "attribute",
            "deprecated",
            "data",
            "plot",
            "warning",
        }:
            print(repr(name), "         ", self.qa)
        return [block_directive]

    def replace_Directive(self, directive: Directive):
        if (directive.domain is not None) or (
            directive.role not in (None, "mod", "class", "func", "meth", "any")
        ):
            # TODO :many of these directive need to be implemented
            if directive.role not in (
                "attr",
                "meth",
                "doc",
                "ref",
                "func",
                "math",
                "mod",
                "class",
                "term",
                "exc",
                "obj",
                "data",
                "sub",
                "program",
                "file",
                "command",
                "sup",
                "rc",  # matplotlib
            ):
                print(directive.role)
            return [directive]
        loc: FrozenSet[str]
        if directive.role not in ["any", None]:
            loc = frozenset()
        else:
            loc = self.local_refs
        r = resolve_(self.qa, self.known_refs, loc, directive.text, rev_aliases=self.rev_aliases)
        # this is now likely incorrect as Ref kind should not be exists,
        # but things like "local", "api", "gallery..."
        ref, exists = r.path, r.kind
        if exists != "missing":
            if exists == "local":
                self.local.append(directive.text)
            else:
                self.total.append((directive.text, ref))
            return [Link(directive.text, r, exists, exists != "missing")]
        return [directive]


def load_one(
    bytes_: bytes, bytes2_: bytes, qa: str, known_refs: FrozenSet[RefInfo] = None
) -> IngestedBlobs:
    data = json.loads(bytes_)
    assert "backrefs" not in data
    # OK to mutate we are the only owners and don't return it.
    data["backrefs"] = json.loads(bytes2_) if bytes2_ else []
    blob = IngestedBlobs.from_json(data)
    # TODO move that one up.
    if known_refs is None:
        known_refs = frozenset()
    blob.process(qa, known_refs=known_refs, aliases={})
    return blob


class Ingester:
    def __init__(self):
        self.ingest_dir = ingest_dir

    def ingest(self, path: Path, check: bool):

        store = Store(self.ingest_dir)
        known_refs, _ = find_all_refs(store)

        nvisited_items = {}
        other_backrefs = {}
        root = None
        meta_path = path / "papyri.json"
        with meta_path.open() as f:
            data = json.loads(f.read())
            version = data["version"]
            logo = data.get("logo", None)
            # long : short
            aliases:Dict[str, str] = data.get("aliases", {})
            root = data.get("module")
            print(len(set(aliases.keys())), len(set(aliases.values())))

        del f

        for p, f1 in progress(
            (path / "module").glob("*.json"),
            description=f"Reading {path.name} doc bundle files ...",
        ):
            qa = f1.name[:-5]
            if check:
                rqa = normalise_ref(qa)
                if rqa != qa:
                    # numpy weird thing
                    print(f"skip {qa}")
                    continue
                assert rqa == qa, f"{rqa} !+ {qa}"
            try:
                with f1.open() as fff:
                    brpath = Path(str(f1)[:-5] + "br")
                    br: Optional[bytes]
                    if brpath.exists():
                        br = brpath.read_bytes()
                    else:
                        br = None
                    nvisited_items[qa] = load_one_uningested(
                        fff.read(), br, qa=qa, known_refs=known_refs, aliases=aliases
                    )
            except Exception as e:
                raise RuntimeError(f"error Reading to {f1}") from e
        del f1

        (self.ingest_dir / root).mkdir(exist_ok=True)
        (self.ingest_dir / root / version).mkdir(exist_ok=True)
        (self.ingest_dir / root / version / "module").mkdir(exist_ok=True)
        known_refs_II = frozenset(nvisited_items.keys())

        # TODO :in progress, crosslink needs version information.
        known_ref_info = frozenset(
            RefInfo(root, version, "api", qa) for qa in known_refs_II
        ).union(known_refs)

        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Cross referencing"
        ):
            for k, v in doc_blob.content.items():
                assert isinstance(v, Section), f"section {k} is not a Section: {v!r}"
            local_refs = frozenset(
                [x[0] for x in doc_blob.content["Parameters"] if x[0]]
                + [x[0] for x in doc_blob.content["Returns"] if x[0]]
            )
            doc_blob.process(qa, known_ref_info, verbose=False, aliases=aliases)
            doc_blob.logo = logo
            for ref in doc_blob.refs:
                hash(known_ref_info)
                hash(local_refs)
                # r = resolve_(qa, known_ref_info, local_refs, ref)
                resolved, exists = ref.reference.path, ref.reference.kind
                # here need to check and load the new files touched.
                if resolved in nvisited_items and ref != qa:
                    nvisited_items[resolved].backrefs.append(qa)
                elif ref != qa and exists == "missing" and "." in ref.reference.path:
                    assert ref.reference.module is None
                    ref_root = ref.reference.path.split(".")[0]
                    if ref_root == root:
                        continue
                    if ref_root == "builtins":
                        continue

                    existing_locations = list(
                        (self.ingest_dir / ref_root).glob(
                            f"*/module/{ge(resolved)}.json"
                        )
                    )
                    # TODO: get the latest, if there are many versions.
                    # assert len(existing_locations) <= 1, existing_locations
                    # pass
                    if not existing_locations:
                        # print("Could not find", resolved, ref, f"({qa})")
                        continue
                    existing_location = existing_locations[0]
                    # existing_location = (
                    #    self.ingest_dir / ref_root / "module" / (resolved + ".json")
                    # )
                    if existing_location.exists():
                        brpath = Path(str(existing_location)[:-5] + ".br")
                        brdata: Optional[bytes]
                        if brpath.exists():
                            brdata = brpath.read_bytes()
                        else:
                            assert False
                            brdata = None
                        try:
                            other_backrefs[resolved] = load_one(
                                existing_location.read_bytes(),
                                brdata,
                                qa=resolved,
                            )
                        except Exception as e:
                            raise type(e)(f"Error in {qa} {existing_location}")
                        other_backrefs[resolved].backrefs.append(qa)
                    elif "/" not in resolved:
                        # TODO figure out this one.
                        phantom_dir = (
                            self.ingest_dir / ref_root / "module" / "__phantom__"
                        )
                        phantom_dir.mkdir(exist_ok=True, parents=True)
                        ph = phantom_dir / (resolved + ".json")
                        if ph.exists():
                            ph_data = json.loads(ph.read_text())

                        else:
                            ph_data = []
                        ph_data.append(qa)
                        ph.write_text(json.dumps(ph_data))
                    else:
                        print(resolved, "not valid reference, skipping.")
            # todo: warning mutation.
            for sa in doc_blob.see_also:
                rev_aliases = {v:k for k,v in aliases.items()}
                r = resolve_(qa, known_ref_info, frozenset(), sa.name.name, rev_aliases, rev_aliases)
                resolved, exists = r.path, r.kind
                if exists == "exists":
                    sa.name.exists = True
                    sa.name.ref = resolved
        (self.ingest_dir / root / version / "assets").mkdir(exist_ok=True)
        for px, f2 in progress(
            (path / "assets").glob("*"),
            description=f"Reading {path.name} image files ...",
        ):
            (self.ingest_dir / root / version / "assets" / f2.name).write_bytes(
                f2.read_bytes()
            )

        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Cleaning double references"
        ):
            # TODO: load backrref from either:
            # 1) previsous version fo the file
            # 2) phantom file if first import (and remove the phantom file?)
            phantom_dir = self.ingest_dir / root / version / "module" / "__phantom__"
            ph = phantom_dir / (qa + ".json")
            if ph.exists():
                ph_data = json.loads(ph.read_text())
            else:
                ph_data = []

            doc_blob.backrefs = list(sorted(set(doc_blob.backrefs + ph_data)))
        for console, path in progress(
            (self.ingest_dir / root / version / "module").glob("*.json"),
            description="cleanig previsous files ....",
        ):
            path.unlink()

        with open(self.ingest_dir / root / version / "papyri.json", "w") as f:
            f.write(json.dumps(aliases))

        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Writing..."
        ):
            # we might update other modules with backrefs
            for k, v in doc_blob.content.items():
                assert isinstance(v, Section), f"section {k} is not a Section: {v!r}"
            mod_root = qa.split(".")[0]
            assert mod_root == root, f"{mod_root}, {root}"
            # TODO : this is wrong, we get version of module we are currently ingesting
            doc_blob.version = version
            js = doc_blob.to_json()
            br = js.pop("backrefs", [])
            try:
                path = self.ingest_dir / mod_root / version / "module" / f"{qa}.json"
                path_br = self.ingest_dir / mod_root / version / "module" / f"{qa}.br"

                with path.open("w") as f:
                    f.write(json.dumps(js, indent=2))
                if path_br.exists():
                    bb = json.loads(path_br.read_text())
                else:
                    bb = []
                path_br.write_text(json.dumps(list(sorted(set(br + bb)))))
            except Exception as e:
                raise RuntimeError(f"error writing to {path}") from e

        for p, (qa, doc_blob) in progress(
            other_backrefs.items(), description="Updating other crossrefs..."
        ):
            # we might update other modules with backrefs
            for k, v in doc_blob.content.items():
                assert isinstance(v, Section), f"section {k} is not a Section: {v!r}"
            mod_root = qa.split(".")[0]
            assert mod_root != root
            js = doc_blob.to_json()
            br = js.pop("backrefs", [])
            try:
                path = (
                    self.ingest_dir
                    / mod_root
                    / doc_blob.version
                    / "module"
                    / f"{qa}.json"
                )
                path_br = (
                    self.ingest_dir
                    / mod_root
                    / doc_blob.version
                    / "module"
                    / f"{qa}.br"
                )
                with path.open("w") as f:
                    f.write(json.dumps(js, indent=2))
                if path_br.exists():
                    bb = json.loads(path_br.read_text())
                else:
                    bb = []
                path_br.write_text(json.dumps(list(sorted(set(br + bb)))))
            except Exception as e:
                raise RuntimeError(f"error writing to {path}") from e


def main(path, check):
    print("Ingesting ", path.name)
    Ingester().ingest(path, check)


def relink():
    store = Store(ingest_dir)
    known_refs, _ = find_all_refs(store)

    
    aliases:Dict[str, str] = {}
    for meta_path in store.glob('*/*/papyri.json'):
        print(meta_path)
        data = json.loads(meta_path.path.read_text())
        aliases.update(data)
    print(len(aliases))
    rev_aliases = {v:k for k,v in aliases.items()}

    builtins.print(
        "Relinking is safe to cancel, but some back references may be broken...."
    )
    builtins.print("Press Ctrl-C to abort...")
    for p, item in progress(
        store.glob("*/*/module/*.json"), description="Relinking..."
    ):
        try:
            data = json.loads(item.path.read_text())
        except Exception as e:
            raise ValueError(str(item)) from e
        qa = item.path.name[:-5]
        data["backrefs"] = []
        doc_blob = IngestedBlobs.from_json(data)
        doc_blob.process(qa, known_refs, aliases=aliases)
        
        for sa in doc_blob.see_also:
            r = resolve_(qa, known_refs, frozenset(), sa.name.name, rev_aliases=rev_aliases)
            resolved, exists = r.path, r.kind
            if exists == "exists":
                sa.name.exists = True
                sa.name.ref = resolved

        data = doc_blob.to_json()
        data.pop("backrefs")
        item.path.write_text(json.dumps(data, indent=2))
