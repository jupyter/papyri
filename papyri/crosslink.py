from __future__ import annotations

import builtins
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

from there import print

from .config import ingest_dir
from .gen import DocBlob, normalise_ref
from .graphstore import GraphStore
from .take2 import (
    Admonition,
    Directive,
    BlockDirective,
    Link,
    Node,
    Param,
    RefInfo,
    Section,
    SeeAlsoItem,
    Code2,
    Token,
)
from .utils import progress

warnings.simplefilter("ignore", UserWarning)


def g_find_all_refs(graph_store):
    o_family = sorted(list(graph_store.glob((None, None, "module", None))))

    # TODO
    # here we can't compute just the dictionary and use frozenset(....values())
    # as we may have multiple version of libraries; this is something that will
    # need to be fixed in the long run
    known_refs = []
    ref_map = {}
    for item in o_family:
        r = RefInfo(item.module, item.version, "module", item.path)
        known_refs.append(r)
        ref_map[r.path] = r
    return frozenset(known_refs), ref_map


def find_all_refs(store):
    if isinstance(store, GraphStore):
        return g_find_all_refs(store)
    assert False

    o_family = sorted(
        list(r for r in store.glob("*/*/module/*") if not r.path.name.endswith(".br"))
    )

    # TODO
    # here we can't compute just the dictionary and use frozenset(....values())
    # as we may have multiple version of libraries; this is something that will
    # need to be fixed in the long run
    known_refs = []
    ref_map = {}
    for item in o_family:
        module, v = item.path.parts[-4:-2]
        r = RefInfo(module, v, "module", item.name)
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
        "qa",
        "arbitrary",
    )

    _content: Dict[str, Section]
    refs: List[RefInfo]
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
    backrefs: List[RefInfo]
    qa: str
    arbitrary: List[Section]

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

    def process(self, known_refs, aliases, verbose=True):
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
        if self.refs is None:
            self.refs = []
        for r in self.refs:
            assert None not in r
        if aliases is None:
            aliases = {}
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

        visitor = DVR(self.qa, known_refs, local_refs, aliases)
        for section in ["Extended Summary", "Summary", "Notes"] + sections_:
            assert section in self.content
            self.content[section] = visitor.visit(self.content[section])
        if (len(visitor.local) or len(visitor.total)) and verbose:
            # TODO: reenable assert len(visitor.local) == 0, f"{visitor.local} | {self.qa}"
            print(f"Newly found {len(visitor.total)} links in {self.qa}:")
            for a, b in visitor.total:
                print("     ", repr(a), "refers to", repr(b))

        self.example_section_data = visitor.visit(self.example_section_data)

        self.arbitrary = [visitor.visit(s) for s in self.arbitrary]

        for d in self.see_also:
            new_desc = []
            for dsc in d.descriptions:
                new_desc.append(visitor.visit(dsc))
            d.descriptions = new_desc
        try:
            for r in visitor._targets:
                assert None not in r, r
            self.refs = list(set(visitor._targets).union(set(self.refs)))
            for r in self.refs:
                assert None not in r
        except Exception as e:
            raise type(e)(self.refs)

    @classmethod
    def from_json(cls, data):
        inst = super().from_json(data)
        inst._freeze()
        return inst


# iii = 0


# @lru_cache(maxsize=100000)
def _into(known_refs: List[Union[RefInfo, str]]) -> Dict[str, RefInfo]:
    # Tuple[FrozenSet[RefInfo], FrozenSet[str]]:
    # global iii
    # iii += 1
    # print("III", iii)

    _map: Dict[str, List[RefInfo]] = defaultdict(lambda: [])
    assert isinstance(known_refs, frozenset)
    # k_path_map = frozenset({k.path for k in known_refs})
    for k in known_refs:
        _map[k.path].append(k)

    _m2 = {}
    for k, v in _map.items():
        cand = list(sorted(v, key=lambda x: x.version))
        assert len(set(c.module for c in cand)) == 1, cand
        _m2[k] = cand[-1]

    return _m2, frozenset(_m2.keys())


@lru_cache
def root_start(root, refs):
    return frozenset(r for r in refs if r.startswith(root))


@lru_cache(10000)
def endswith(end, refs):
    return frozenset(r for r in refs if r.endswith(end))


_cache = {}


def resolve_(
    qa: str,
    known_refs: FrozenSet[RefInfo],
    local_refs: FrozenSet[str],
    ref: str,
    rev_aliases=None,
) -> RefInfo:
    # RefInfo(module, version, kind, path)
    hk = hash(known_refs)
    hash(local_refs)
    if rev_aliases is None:
        rev_aliases = {}
    if ref in rev_aliases:
        new_ref = rev_aliases[ref]
        # print(f'now looking for {new_ref} instead of {ref}')
        assert new_ref not in rev_aliases, "would loop...."
        # TODOlikely can drop rev_aliases here
        res = resolve_(qa, known_refs, local_refs, new_ref, rev_aliases)
        return res

    assert isinstance(ref, str), ref

    # TODO: LRU Cache seem to have speed problem here; and get slow while this should be just fine.
    # this seem to be due to the fact that even if the hash is the same this still needs to compare the objects, as
    # those may have been muted.
    if hk not in _cache:
        _cache[hk] = _into(known_refs)

    k_path_map, keyset = _cache[hk]

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
        # get the more recent.
        # stuff = {k for k in known_refs if k.path == ref}
        # c2 = list(sorted(stuff, key=lambda x: x.version))[-1]
        # assert isinstance(c2, RefInfo), c2
        # assert k_path_map[ref] == c2
        return k_path_map[ref]
    else:
        if ref.startswith("."):
            if (found := qa + ref) in k_path_map:
                return k_path_map[found]
            else:
                root = qa.split(".")[0]
                sub1 = root_start(root, keyset)
                subset = endswith(ref, sub1)
                if len(subset) == 1:
                    return k_path_map[next(iter(subset))]
                    # return RefInfo(None, None, "exists", next(iter(subset)))
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
                return k_path_map[attempt]

    q0 = qa.split(".")[0]
    rs = root_start(q0, keyset)
    attempts = [q for q in rs if (ref in q)]
    if len(attempts) == 1:
        # return RefInfo(None, None, "exists", attempts[0])
        return k_path_map[attempts[0]]

    return RefInfo(None, None, "missing", ref)


def load_one_uningested(
    bytes_: bytes, bytes2_: Optional[bytes], qa, known_refs, aliases
) -> IngestedBlobs:
    """
    Load the json from a DocBlob and make it an ingested blob.
    """
    data = json.loads(bytes_)

    old_data = DocBlob.from_json(data)
    assert hasattr(old_data, "arbitrary")

    blob = IngestedBlobs()
    blob.qa = qa
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
    # blob.refs = list(sorted(set(blob.refs)))
    blob.refs = []

    for r in blob.refs:
        assert None not in r

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

    acc1 = []
    for sec in blob.arbitrary:
        acc1.append(visitor.visit(sec))
    blob.arbitrary = acc1

    blob.process(known_refs=known_refs, aliases=aliases, verbose=False)
    # if targets:
    #    print("LL", len(targets))

    # if blob.refs:
    #    new_refs: List[RefInfo] = []
    #    kind = "exists"
    #    for value in blob.refs:
    #        assert isinstance(value, str)
    #        r = resolve_(qa, known_refs, frozenset(), value)
    #        if None not in r:
    #            new_refs.append(r)

    #    blob.refs = new_refs
    # blob.refs = list(targets)
    # if blob.refs:
    #    print("BLOB REFS:", blob.refs)
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
                "BlockQuote",
                "BulletList",
                "Directive",
                "SeeAlsoItems",
                "Code2",
            ]:
                return [node]
            elif name in ["Text"]:
                assert False, "Text still present"
            else:
                new_children = []
                if not hasattr(node, "children"):
                    raise ValueError(f"{node.__class__} has no children {node}")
                for c in node.children:
                    assert c is not None, f"{node=} has a None child"
                    assert isinstance(c, Node), node
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
        self.rev_aliases = {v: k for k, v in aliases.items()}
        self._targets = set()

    def replace_BlockDirective(self, block_directive: BlockDirective):
        block_directive.children = [self.visit(c) for c in block_directive.children]
        if block_directive.directive_name in ["warning", "notes"]:
            title = None
            args0 = block_directive.args0
            args0 = [a.strip() for a in args0 if a.strip()]
            if args0:
                # assert len(args0) == 1
                # TODO: dont' allow admonition on first line.
                # print(
                #    "ADM!!",
                #    self.qa,
                #    "does title block adm",
                #    repr(args0),
                #    repr(block_directive.children),
                # )
                title = args0[0]

            assert block_directive.children is not None, block_directive
            return [
                Admonition(
                    block_directive.directive_name, title, block_directive.children
                )
            ]
        if block_directive.directive_name in [
            "deprecated",
            "code",
            "versionchanged",
            "autosummary",
            "note",
            "warning",
            "math",
            "versionadded",
            "attribute",
            "hint",
            "plot",
            "seealso",
            "moduleauthor",
            "data",
            "WARNING",
            "currentmodule",
            "important",
            "code-block",
            "image",
            "rubric",
            "inheritance-diagram",
            "table",
        ]:
            return [block_directive]
        print(block_directive.directive_name, self.qa)

        return [block_directive]

    def _resolve(self, loc, text):
        assert isinstance(text, str)
        return resolve_(
            self.qa, self.known_refs, loc, text, rev_aliases=self.rev_aliases
        )

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
                print("TODO role:", directive.role)
            return [directive]
        loc: FrozenSet[str]
        if directive.role not in ["any", None]:
            loc = frozenset()
        else:
            loc = self.local_refs
        r = self._resolve(loc, directive.text)
        # this is now likely incorrect as Ref kind should not be exists,
        # but things like "local", "api", "gallery..."
        ref, exists = r.path, r.kind
        if exists != "missing":
            if exists == "local":
                self.local.append(directive.text)
            else:
                self.total.append((directive.text, ref))
            if r.kind != "local":
                assert None not in r, r
                self._targets.add(r)
            return [Link(directive.text, r, exists, exists != "missing")]
        return [directive]


class DVR(DirectiveVisiter):
    def replace_Code(self, code):
        """
        Here we'll crawl example data and convert code entries so that each token contain a link to the object they
        refered to.
        """
        new_entries = []
        for entry in code.entries:
            # TODO
            if entry[1] and entry[1].strip():
                # print(entry[1])
                r = self._resolve(frozenset(), entry[1])
                if r.kind == "api":
                    self._targets.add(r)
                    new_entries.append(
                        Token(
                            Link(
                                str(entry[0]),
                                r,
                                "module",
                                True,
                            ),
                            entry[2],
                        )
                    )
                    continue
            new_entries.append(
                Token(str(entry[0]), entry[2]),
            )

        return [Code2(new_entries, code.out, code.ce_status)]


def load_one(
    bytes_: bytes, bytes2_: bytes, known_refs: FrozenSet[RefInfo] = None, strict=False
) -> IngestedBlobs:
    data = json.loads(bytes_)
    assert "backrefs" not in data
    # OK to mutate we are the only owners and don't return it.
    data["backrefs"] = json.loads(bytes2_) if bytes2_ else []
    blob = IngestedBlobs.from_json(data)
    # TODO move that one up.
    if known_refs is None:
        known_refs = frozenset()
    if not strict:
        targets = blob.process(known_refs=known_refs, aliases=None)
        if targets:
            print("OA", len(targets))
    return blob


class Ingester:
    def __init__(self):
        self.ingest_dir = ingest_dir
        self.gstore = GraphStore(self.ingest_dir)

    def ingest(self, path: Path, check: bool):

        gstore = self.gstore

        known_refs, _ = find_all_refs(gstore)

        nvisited_items = {}

        ###

        meta_path = path / "papyri.json"
        data = json.loads(meta_path.read_text())
        version = data["version"]
        root = data["module"]
        logo = data.get("logo", None)
        # long : short
        aliases: Dict[str, str] = data.get("aliases", {})
        rev_aliases = {v: k for k, v in aliases.items()}


        for _, fe in progress(
            (path / "examples/").glob("*"), description=f"Reading {path.name} Examples"
        ):
            s = Section.from_json(json.loads(fe.read_text()))
            gstore.put(
                (root, version, "examples", fe.name),
                json.dumps(s.to_json()).encode(),
                [],
            )

        for _, f1 in progress(
            (path / "module").glob("*"),
            description=f"Reading {path.name} doc bundle files ...",
        ):
            assert f1.name.endswith(".json")
            qa = f1.name[:-5]
            if check:
                rqa = normalise_ref(qa)
                if rqa != qa:
                    # numpy weird thing
                    print(f"skip {qa}")
                    continue
                assert rqa == qa, f"{rqa} !+ {qa}"
            try:
                nvisited_items[qa] = load_one_uningested(
                    f1.read_text(),
                    None,
                    qa=qa,
                    known_refs=known_refs,
                    aliases=aliases,
                )
                assert hasattr(nvisited_items[qa], "arbitrary")
            except Exception as e:
                raise RuntimeError(f"error Reading to {f1}") from e

        known_refs_II = frozenset(nvisited_items.keys())

        # TODO :in progress, crosslink needs version information.
        known_ref_info = frozenset(
            RefInfo(root, version, "module", qa) for qa in known_refs_II
        ).union(known_refs)

        for _, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Cross referencing"
        ):
            refs = doc_blob.process(known_ref_info, verbose=False, aliases=aliases)
            doc_blob.logo = logo
            # todo: warning mutation.
            for sa in doc_blob.see_also:
                r = resolve_(
                    qa,
                    known_ref_info,
                    frozenset(),
                    sa.name.name,
                    rev_aliases=rev_aliases,
                )
                resolved, exists = r.path, r.kind
                if exists == "exists":
                    sa.name.exists = True
                    sa.name.ref = resolved
        for _, f2 in progress(
            (path / "assets").glob("*"),
            description=f"Reading {path.name} image files ...",
        ):
            gstore.put((root, version, "assets", f2.name), f2.read_bytes(), [])

        gstore.put((root, version, "papyri.json"), json.dumps(aliases).encode(), [])

        for _, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Writing..."
        ):
            # we might update other modules with backrefs
            for k, v in doc_blob.content.items():
                assert isinstance(v, Section), f"section {k} is not a Section: {v!r}"
            mod_root = qa.split(".")[0]
            assert mod_root == root, f"{mod_root}, {root}"
            doc_blob.version = version
            assert hasattr(doc_blob, "arbitrary")
            js = doc_blob.to_json()
            refs = [
                (b["module"], b["version"], b["kind"], b["path"])
                for b in js.get("refs", [])
            ]
            for r in refs:
                assert None not in r
            try:
                key = (mod_root, version, "module", qa)
                assert mod_root is not None
                assert version is not None
                assert None not in key
                gstore.put(
                    key,
                    json.dumps(js, indent=2).encode(),
                    refs,
                )

            except Exception as e:
                raise RuntimeError(f"error writing to {path}") from e


    def relink(self):
        gstore = self.gstore
        known_refs, _ = find_all_refs(gstore)
        aliases: Dict[str, str] = {}
        for key in gstore.glob((None, None, "papyri.json")):
            data = gstore.get(key)
            data = json.loads(data)
            aliases.update(data)
        rev_aliases = {v: k for k, v in aliases.items()}

        builtins.print(
            "Relinking is safe to cancel, but some back references may be broken...."
        )
        builtins.print("Press Ctrl-C to abort...")
        for p, key in progress(
            gstore.glob((None, None, "module", None)), description="Relinking..."
        ):
            try:
                data = json.loads(gstore.get(key))
            except Exception as e:
                raise ValueError(str(key)) from e
            qa = key[-1]
            data["backrefs"] = []
            try:
                doc_blob = IngestedBlobs.from_json(data)
            except Exception as e:
                raise type(e)(key)
            doc_blob.process(known_refs, aliases=aliases)

            for sa in doc_blob.see_also:
                r = resolve_(
                    qa, known_refs, frozenset(), sa.name.name, rev_aliases=rev_aliases
                )
                resolved, exists = r.path, r.kind
                if exists == "exists":
                    sa.name.exists = True
                    sa.name.ref = resolved
            data = doc_blob.to_json()
            data.pop("backrefs")
            refs = [
                (b["module"], b["version"], b["kind"], b["path"])
                for b in data.get("refs", [])
            ]
            gstore.put(key, json.dumps(data).encode(), refs)


def main(path, check):
    builtins.print("Ingesting", path.name, "...")
    from time import perf_counter

    now = perf_counter()
    Ingester().ingest(path, check)
    delta = perf_counter() - now

    builtins.print(f"Ingesting {path.name} done in {delta:0.2f}s")


def relink():
    Ingester().relink()
