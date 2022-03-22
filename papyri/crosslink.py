from __future__ import annotations

import builtins
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Any

from rich.logging import RichHandler
import cbor2
from there import print

from .config import ingest_dir
from .gen import DocBlob, normalise_ref
from .graphstore import GraphStore, Key
from .take2 import (
    Node,
    Param,
    RefInfo,
    Fig,
    Section,
    SeeAlsoItem,
    Signature,
    encoder,
    register,
)
from .tree import DVR, DirectiveVisiter, resolve_, TreeVisitor
from .utils import progress, dummy_progress


warnings.simplefilter("ignore", UserWarning)


FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("papyri")


def find_all_refs(
    graph_store: GraphStore,
) -> Tuple[FrozenSet[RefInfo], Dict[str, RefInfo]]:
    assert isinstance(graph_store, GraphStore)
    o_family = sorted(list(graph_store.glob((None, None, "module", None))))

    # TODO
    # here we can't compute just the dictionary and use frozenset(....values())
    # as we may have multiple version of lisbraries; this is something that will
    # need to be fixed in the long run
    known_refs = []
    ref_map = {}
    for item in o_family:
        r = RefInfo(item.module, item.version, "module", item.path)
        known_refs.append(r)
        ref_map[r.path] = r
    return frozenset(known_refs), ref_map


@register(4010)
@dataclass
class IngestedBlobs(Node):

    __slots__ = (
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
    ordered_sections: List[str]
    item_file: Optional[str]
    item_line: Optional[int]
    item_type: Optional[str]
    aliases: List[str]
    example_section_data: Section
    see_also: List[SeeAlsoItem]  # see also data
    version: str
    signature: Signature
    references: Optional[List[str]]
    qa: str
    arbitrary: List[Section]

    __isfrozen = False

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._content = kwargs.pop("_content", None)
        self.example_section_data = kwargs.pop("example_section_data", None)
        self.refs = kwargs.pop("refs", None)
        self.ordered_sections = kwargs.pop("ordered_sections", None)
        self.item_file = kwargs.pop("item_file", None)
        self.item_line = kwargs.pop("item_line", None)
        self.item_type = kwargs.pop("item_type", None)
        self.aliases = kwargs.pop("aliases", [])
        self.see_also = kwargs.pop("see_also", None)
        self.version = kwargs.pop("version", None)
        self.signature = kwargs.pop("signature", None)
        self.references = kwargs.pop("references", None)
        assert "logo" not in kwargs
        self.qa = kwargs.pop("qa", None)
        self.arbitrary = kwargs.pop("arbitrary", None)
        if self.arbitrary:
            for a in self.arbitrary:
                assert isinstance(a, Section), a
        assert not kwargs, kwargs
        assert not args, args
        self._freeze()

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

    def process(
        self, known_refs, aliases: Optional[Dict[str, str]], verbose=True
    ) -> None:
        """
        Process a doc blob, to find all local and nonlocal references.
        """
        assert isinstance(known_refs, frozenset)
        assert self._content is not None
        _local_refs: List[List[str]] = []
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

            _local_refs = _local_refs + [
                [u.strip() for u in x[0].split(",")]
                for x in self.content.get(s, [])
                if isinstance(x, Param)
            ]

        def flat(l):
            return [y for x in l for y in x]

        local_refs = frozenset(flat(_local_refs))

        assert self.version not in ("", "??"), self.version
        visitor = DVR(self.qa, known_refs, local_refs, aliases, version=self.version)
        for section in ["Extended Summary", "Summary", "Notes"] + sections_:
            if section not in self.content:
                continue
            assert section in self.content
            self.content[section] = visitor.visit(self.content[section])
        if (len(visitor.local) or len(visitor.total)) and verbose:
            # TODO: reenable assert len(visitor.local) == 0, f"{visitor.local} | {self.qa}"
            log.info("Newly found %s links in %s", len(visitor.total), repr(self.qa))
            for a, b in visitor.total:
                log.info("     %s refers to %s", repr(a), repr(b))

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

    def from_json(cls, data):
        assert False




# iii = 0


def load_one_uningested(
    bytes_: bytes, bytes2_: Optional[bytes], qa, known_refs, aliases, *, version
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
    assert bytes2_ is None
    blob.version = data.pop("version", version)
    assert blob.version == version

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
            for x in blob.content.get(s, [])
            if isinstance(x, Param)
        ]

    def flat(l) -> List[str]:
        return [y for x in l for y in x]

    local_refs: FrozenSet[str] = frozenset(flat(_local_refs))

    visitor = DirectiveVisiter(qa, frozenset(), local_refs, aliases=aliases)
    for section in ["Extended Summary", "Summary", "Notes"] + sections_:
        if section in blob.content:
            blob.content[section] = visitor.visit(blob.content[section])

    acc1 = []
    for sec in blob.arbitrary:
        acc1.append(visitor.visit(sec))
    blob.arbitrary = acc1

    assert blob.version not in ("", "??"), blob.version

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


def load_one(
    bytes_: bytes, bytes2_: bytes, known_refs: FrozenSet[RefInfo] = None, strict=False
) -> IngestedBlobs:
    assert isinstance(bytes_, bytes), bytes_
    blob = encoder.decode(bytes_)
    assert isinstance(blob, IngestedBlobs)
    # TODO move that one up.
    if known_refs is None:
        known_refs = frozenset()
    if not strict:
        blob.process(known_refs=known_refs, aliases=None)
    return blob


class Ingester:
    def __init__(self, dp):
        self.ingest_dir = ingest_dir
        self.gstore = GraphStore(self.ingest_dir)
        self.progress = dummy_progress if dp else progress

    def _ingest_narrative(self, path, gstore: GraphStore) -> None:

        for _console, document in self.progress(
            (path / "docs").glob("*"), description=f"{path.name} Reading narrative docs"
        ):
            doc = load_one_uningested(
                document.read_text(),
                None,
                qa=document.name,
                known_refs=frozenset(),
                aliases={},
                version=None,
            )
            ref = document.name

            module, version = path.name.split("_")
            key = Key(module, "1.22.1", "docs", ref)
            doc.logo = ""
            doc.version = version
            doc.validate()
            js = doc.to_json()
            assert "backrefs" not in js
            gstore.put(
                key,
                cbor2.dumps(js),
                # json.dumps(js, indent=2).encode(),
                [],
            )

    def _ingest_examples(
        self, path: Path, gstore: GraphStore, known_refs, aliases, version, root
    ):

        for _, fe in self.progress(
            (path / "examples/").glob("*"), description=f"{path.name} Reading Examples"
        ):
            s = Section.from_json(json.loads(fe.read_bytes()))
            visitor = DVR(
                "TBD, supposed to be QA", known_refs, set(), aliases, version=version
            )
            s_code = visitor.visit(s)
            refs = list(map(lambda s: Key(*s), visitor._targets))
            gstore.put(
                Key(root, version, "examples", fe.name),
                encoder.encode(s_code),
                refs,
            )

    def _ingest_assets(self, path, root, version, aliases, gstore):
        for _, f2 in self.progress(
            (path / "assets").glob("*"),
            description=f"{path.name} Reading image files ...",
        ):
            gstore.put(Key(root, version, "assets", f2.name), f2.read_bytes(), [])

        gstore.put(
            Key(root, version, "meta", "papyri.json"),
            cbor2.dumps(aliases),
            # json.dumps(aliases, indent=2).encode(),
            [],
        )

    def ingest(self, path: Path, check: bool) -> None:

        gstore = self.gstore

        known_refs, _ = find_all_refs(gstore)

        nvisited_items = {}

        ###

        meta_path = path / "papyri.json"
        data = json.loads(meta_path.read_text())
        version = data["version"]
        root = data["module"]
        # long : short
        aliases: Dict[str, str] = data.get("aliases", {})
        rev_aliases = {v: k for k, v in aliases.items()}
        meta = {k: v for k, v in data.items() if k != "aliases"}

        gstore.put_meta(root, version, encoder.encode(meta))

        self._ingest_examples(path, gstore, known_refs, aliases, version, root)
        self._ingest_assets(path, root, version, aliases, gstore)
        self._ingest_narrative(path, gstore)

        for _, f1 in self.progress(
            (path / "module").glob("*"),
            description=f"{path.name} Reading doc bundle files ...",
        ):
            assert f1.name.endswith(".json")
            qa = f1.name[:-5]
            if check:
                rqa = normalise_ref(qa)
                if rqa != qa:
                    # numpy weird thing
                    print(f"skip {qa=}, {rqa=}")
                    continue
                assert rqa == qa, f"{rqa} !+ {qa}"
            try:
                # TODO: version issue
                nvisited_items[qa] = load_one_uningested(
                    f1.read_text(),
                    None,
                    qa=qa,
                    known_refs=known_refs,
                    aliases=aliases,
                    version=version,
                )
                assert hasattr(nvisited_items[qa], "arbitrary")
            except Exception as e:
                raise RuntimeError(f"error Reading to {f1}") from e

        known_refs_II = frozenset(nvisited_items.keys())

        # TODO :in progress, crosslink needs version information.
        known_ref_info = frozenset(
            RefInfo(root, version, "module", qa) for qa in known_refs_II
        ).union(known_refs)

        for _, (qa, doc_blob) in self.progress(
            nvisited_items.items(), description=f"{path.name} Cross referencing"
        ):
            doc_blob.process(known_ref_info, verbose=False, aliases=aliases)
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
                if exists == "module":
                    sa.name.exists = True
                    sa.name.ref = resolved

        for _, (qa, doc_blob) in self.progress(
            nvisited_items.items(), description=f"{path.name} Validating..."
        ):
            for k, v in doc_blob.content.items():
                assert isinstance(v, Section), f"section {k} is not a Section: {v!r}"
            try:
                doc_blob.validate()
            except Exception as e:
                raise type(e)(f"from {qa}")
            mod_root = qa.split(".")[0]
            assert mod_root == root, f"{mod_root}, {root}"
        for _, (qa, doc_blob) in self.progress(
            nvisited_items.items(), description=f"{path.name} Writing..."
        ):
            # for qa, doc_blob in nvisited_items.items():
            # we might update other modules with backrefs
            doc_blob.version = version
            assert hasattr(doc_blob, "arbitrary")

            # TODO: FIX
            # when walking the tree of figure we can't properly crosslink
            # as we don't know the version number.
            # fix it at serialisation time.
            forward_refs = []
            for rq in doc_blob.refs:
                assert rq.version != "??"
                assert None not in rq
                forward_refs.append(Key(*rq))

            try:
                key = Key(mod_root, version, "module", qa)
                assert mod_root is not None
                assert version is not None
                assert None not in key
                gstore.put(
                    key,
                    encoder.encode(doc_blob),
                    forward_refs,
                )

            except Exception as e:
                raise RuntimeError(f"error writing to {path}") from e

    def relink(self) -> None:

        gstore = self.gstore
        known_refs, _ = find_all_refs(gstore)
        aliases: Dict[str, str] = {}
        for key in gstore.glob((None, None, "meta", "papyri.json")):
            aliases.update(cbor2.loads(gstore.get(key)))

        rev_aliases = {v: k for k, v in aliases.items()}

        builtins.print(
            "Relinking is safe to cancel, but some back references may be broken...."
        )
        builtins.print("Press Ctrl-C to abort...")

        visitor = TreeVisitor({RefInfo, Fig})
        for _, key in self.progress(
            gstore.glob((None, None, "module", None)), description="Relinking..."
        ):
            try:
                data, back, forward = gstore.get_all(key)
            except Exception as e:
                raise ValueError(str(key)) from e
            try:
                doc_blob = encoder.decode(data)
                assert isinstance(doc_blob, IngestedBlobs)
                # if res:
                # print("Refinfos...", len(res))
            except Exception as e:
                raise type(e)(key)
            assert doc_blob.content is not None, data
            doc_blob.process(known_refs, aliases=aliases)

            # TODO: Move this into process ?
            res: Dict[Any, List[Any]] = {}
            for sec in (
                list(doc_blob.content.values())
                + [doc_blob.example_section_data]
                + doc_blob.arbitrary
                + doc_blob.see_also
            ):
                for k, v in visitor.generic_visit(sec).items():
                    res.setdefault(k, []).extend(v)

            assets_II = {
                Key(key.module, key.version, "assets", f.value)
                for f in res.get(Fig, [])
            }
            sr = set(
                [Key(*r) for r in res.get(RefInfo, []) if r.kind != "local"]
            ).union(assets_II)

            for sa in doc_blob.see_also:
                if sa.name.exists:
                    continue
                r = resolve_(
                    key.path,
                    known_refs,
                    frozenset(),
                    sa.name.name,
                    rev_aliases=rev_aliases,
                )
                resolved, exists = r.path, r.kind
                if exists == "module":
                    print("unresolved ok...", r, key)
                    sa.name.exists = True
                    sa.name.ref = resolved

            # end todo

            data = encoder.encode(doc_blob)
            if set(sr) != set(forward):
                gstore.put(key, data, [Key(*x) for x in sr])

        for _, key in progress(
            gstore.glob((None, None, "examples", None)),
            description="Relinking Examples...",
        ):
            s = encoder.decode(gstore.get(key))
            assert isinstance(s, Section), (s, key)
            dvr = DVR("TBD, supposed to be QA", known_refs, set(), aliases, version="?")
            s_code = dvr.visit(s)
            refs = [Key(*x) for x in dvr._targets]
            gstore.put(
                key,
                encoder.encode(s_code),
                refs,
            )


def main(path, check, *, dummy_progress):
    """
    Parameters
    ----------
    dummy_progress : bool
        whether to use a dummy progress bar instead of the rich one.
        Usefull when dropping into PDB.
        To be implemented. See gen step.
    check : <Insert Type here>
        <Multiline Description Here>
    path : <Insert Type here>
        <Multiline Description Here>
    """
    builtins.print("Ingesting", path.name, "...")
    from time import perf_counter

    now = perf_counter()

    assert path.exists(), f"{path} does not exists"
    assert path.is_dir(), f"{path} is not a directory"
    Ingester(dp=dummy_progress).ingest(path, check)
    delta = perf_counter() - now

    builtins.print(f"{path.name} Ingesting done in {delta:0.2f}s")


def relink(dummy_progress):
    Ingester(dp=dummy_progress).relink()
