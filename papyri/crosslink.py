from __future__ import annotations

import builtins
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from rich.logging import RichHandler
from there import print

from .config import ingest_dir
from .gen import DocBlob, normalise_ref
from .graphstore import GraphStore, Key
from .take2 import Node, Param, RefInfo, Section, SeeAlsoItem, Signature
from .tree import DVR, DirectiveVisiter, resolve_
from .utils import progress, dummy_progress

warnings.simplefilter("ignore", UserWarning)


FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("papyri")


def g_find_all_refs(graph_store) -> Tuple[FrozenSet[RefInfo], Dict[str, RefInfo]]:
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


def find_all_refs(store) -> Tuple[FrozenSet[RefInfo], Dict[str, RefInfo]]:
    if isinstance(store, GraphStore):
        return g_find_all_refs(store)

    # Used in render. need to split somewhere else.

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
    refs: List[RefInfo]
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
    logo: Optional[str]
    backrefs: List[RefInfo]
    qa: str
    arbitrary: List[Section]

    __isfrozen = False

    @classmethod
    def _deserialise(cls, **kwargs):
        # print("will deserialise", cls)
        try:
            instance = cls._instance()
        except Exception as e:
            raise type(e)(f"Error deserialising {cls}, {kwargs})") from e
        assert "_content" in kwargs
        assert kwargs["_content"] is not None
        for k, v in kwargs.items():
            setattr(instance, k, v)
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backrefs = []
        self._content = kwargs.pop("_content", None)
        self.example_section_data = kwargs.pop("example_section_data", None)
        self.refs = kwargs.pop("refs", None)
        self.ordered_sections = kwargs.pop("ordered_sections", None)
        self.item_file = kwargs.pop("item_file", None)
        self.item_line = kwargs.pop("item_line", None)
        self.item_type = kwargs.pop("item_type", None)
        self.aliases = []
        assert not kwargs, kwargs

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
        """
        Process a doc blob, to find all local and nonlocal references.
        """
        assert isinstance(known_refs, frozenset)
        assert self._content is not None
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
                for x in self.content.get(s, [])
                if isinstance(x, Param)
            ]

        def flat(l):
            return [y for x in l for y in x]

        local_refs = frozenset(flat(local_refs))

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

    @classmethod
    def from_json(cls, data):
        inst = super().from_json(data)
        inst._freeze()
        return inst


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
    if bytes2_ is not None:
        backrefs = json.loads(bytes2_)
    else:
        backrefs = []
    blob.backrefs = backrefs
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
    def __init__(self, dp):
        self.ingest_dir = ingest_dir
        self.gstore = GraphStore(self.ingest_dir)
        self.progress = dummy_progress if dp else progress

    def _ingest_narrative(self, path, gstore):

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
            del js["backrefs"]
            gstore.put(
                key,
                json.dumps(js, indent=2).encode(),
                [],
            )

    def _ingest_examples(self, path: Path, gstore, known_refs, aliases, version, root):

        for _, fe in self.progress(
            (path / "examples/").glob("*"), description=f"{path.name} Reading Examples"
        ):
            s = Section.from_json(json.loads(fe.read_text()))
            visitor = DVR(
                "TBD, supposed to be QA", known_refs, {}, aliases, version=version
            )
            s_code = visitor.visit(s)
            refs = list(map(tuple, visitor._targets))
            gstore.put(
                Key(root, version, "examples", fe.name),
                json.dumps(s_code.to_json(), indent=2).encode(),
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
            json.dumps(aliases, indent=2).encode(),
            [],
        )

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
                if exists == "module":
                    sa.name.exists = True
                    sa.name.ref = resolved

        for _, (qa, doc_blob) in self.progress(
            nvisited_items.items(), description=f"{path.name} Writing..."
        ):
            # for qa, doc_blob in nvisited_items.items():
            # we might update other modules with backrefs
            for k, v in doc_blob.content.items():
                assert isinstance(v, Section), f"section {k} is not a Section: {v!r}"
            mod_root = qa.split(".")[0]
            assert mod_root == root, f"{mod_root}, {root}"
            doc_blob.version = version
            assert hasattr(doc_blob, "arbitrary")

            try:
                doc_blob.validate()
            except Exception as e:
                raise type(e)(f"from {qa}")
            js = doc_blob.to_json()
            del js["backrefs"]

            # TODO: FIX
            # when walking the tree of figure we can't properly crosslink
            # as we don't know the version number.
            # fix it at serialisation time.
            rr = []
            for rq in js["refs"]:
                assert rq["version"] != "??"
                if rq["version"] == "??":
                    rq["version"] = version
                rr.append(rq)
            js["refs"] = rr

            refs = [
                (b["module"], b["version"], b["kind"], b["path"])
                for b in js.get("refs", [])
            ]
            for xr in refs:
                assert None not in xr
            try:
                key = Key(mod_root, version, "module", qa)
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
        for key in gstore.glob((None, None, "meta", "papyri.json")):
            aliases.update(json.loads(gstore.get(key)))

        rev_aliases = {v: k for k, v in aliases.items()}

        builtins.print(
            "Relinking is safe to cancel, but some back references may be broken...."
        )
        builtins.print("Press Ctrl-C to abort...")

        for _, key in self.progress(
            gstore.glob((None, None, "module", None)), description="Relinking..."
        ):
            try:
                data = json.loads(gstore.get(key))
            except Exception as e:
                raise ValueError(str(key)) from e
            data["backrefs"] = []
            try:
                doc_blob = IngestedBlobs.from_json(data)
            except Exception as e:
                raise type(e)(key)
            assert doc_blob.content is not None, data
            doc_blob.process(known_refs, aliases=aliases)

            # TODO: Move this into process ?

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
                    sa.name.exists = True
                    sa.name.ref = resolved

            # end todo

            data = doc_blob.to_json()
            data.pop("backrefs")
            refs = [
                (b["module"], b["version"], b["kind"], b["path"])
                for b in data.get("refs", [])
            ]
            gstore.put(key, json.dumps(data, indent=2).encode(), refs)

        for _, key in progress(
            gstore.glob((None, None, "examples", None)),
            description="Relinking Examples...",
        ):
            s = Section.from_json(json.loads(gstore.get(key)))
            visitor = DVR(
                "TBD, supposed to be QA", known_refs, {}, aliases, version="?"
            )
            s_code = visitor.visit(s)
            refs = list(map(tuple, visitor._targets))
            gstore.put(
                key,
                json.dumps(s_code.to_json(), indent=2).encode(),
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
    print(dummy_progress)
    Ingester(dp=dummy_progress).relink()
