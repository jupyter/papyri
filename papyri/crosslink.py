import dataclasses
import json
import warnings
from functools import lru_cache
from pathlib import Path

from there import print

from .config import ingest_dir
from .gen import DocBlob, normalise_ref
from .take2 import Lines, Link, Math, Node, Paragraph, Ref, Section, SeeAlsoItem
from .take2 import main as t2main
from .take2 import make_block_3
from .utils import progress

from pygments import lex
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

warnings.simplefilter("ignore", UserWarning)


@dataclass(frozen=True)
class RefInfo:
    module: str
    version: str
    kind: str
    path: str


def paragraph(lines) -> List[Tuple[str, Any]]:
    """
    return container of (type, obj)
    """
    p = Paragraph.parse_lines(lines)
    acc = []
    for c in p.children:
        if type(c).__name__ == "Directive":
            if c.role == "math":
                acc.append(Math(c.value))
            else:
                acc.append(c)
        else:
            acc.append(c)
    p.children = acc
    return p


def P2(lines) -> List[Node]:
    assert isinstance(lines, list)
    for l in lines:
        if isinstance(l, str):
            assert "\n" not in l
        else:
            assert "\n" not in l._line
    assert lines, lines
    blocks_data = t2main("\n".join(lines))

    # for pre_blank_lines, blank_lines, post_black_lines in blocks_data:
    for block in blocks_data:
        assert not block.__class__.__name__ == "Block"
    return blocks_data


def paragraphs(lines) -> List[Any]:
    assert isinstance(lines, list)
    for l in lines:
        if isinstance(l, str):
            assert "\n" not in l
        else:
            assert "\n" not in l._line
    blocks_data = make_block_3(Lines(lines))
    acc = []

    # blocks_data = t2main("\n".join(lines))

    # for pre_blank_lines, blank_lines, post_black_lines in blocks_data:
    for pre_blank_lines, blank_lines, post_black_lines in blocks_data:
        # pre_blank_lines = block.lines
        # blank_lines = block.wh
        # post_black_lines = block.ind
        if pre_blank_lines:
            acc.append(paragraph([x._line for x in pre_blank_lines]))
        ## definitively wrong but will do for now, should likely be verbatim, or recurse ?
        if post_black_lines:
            acc.append(paragraph([x._line for x in post_black_lines]))
        # print(block)
    return acc


def processed_example_data(example_section_data, qa):
    new_example_section_data = Section()
    for in_out in example_section_data:
        type_ = in_out.__class__.__name__
        # color examples with pygments classes
        if type_ == "Code":
            in_ = in_out.entries
            if len(in_[0]) == 2:
                text = "".join([x for x, y in in_])
                classes = get_classes(text)
                in_out.entries = [ii + (cc,) for ii, cc in zip(in_, classes)]
        new_example_section_data.append(in_out)

    return new_example_section_data



def get_classes(code):
    list(lex(code, PythonLexer()))
    FMT = HtmlFormatter()
    classes = [FMT.ttype2class.get(x) for x, y in lex(code, PythonLexer())]
    classes = [c if c is not None else "" for c in classes]
    return classes


class IngestedBlobs(DocBlob):

    __slots__ = ("backrefs", "see_also", "version", "logo")

    see_also: List[SeeAlsoItem]  # see also data
    version: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backrefs = []

    def slots(self):
        return super().slots() + ["backrefs", "see_also", "version", "logo"]

    @classmethod
    def from_json(cls, data, qa=None):
        instance = cls()
        for k, v in data.items():
            setattr(instance, k, v)
        if instance._content is None:
            instance._content = {}

        # instance._content["Parameters"] = [
        #    Parameter(a, b, c) for (a, b, c) in instance._content.get("Parameters", [])
        # ]

        for it in (
            "Returns",
            "Yields",
            "Extended Summary",
            "Receives",
            "Other Parameters",
            "Raises",
            "Warns",
            "Warnings",
            "See Also",
            "Notes",
            "References",
            "Examples",
            "Attributes",
            "Methods",
        ):
            pass

        instance.see_also = [SeeAlsoItem.from_json(x) for x in data.pop("see_also", [])]

        # Todo: remove this; hopefully the logic from load_one_uningested
        # load a DocBlob instaead of un IngestedDocBlob
        # or more likely the paragraph parsing is made in Gen.
        assert isinstance(instance.example_section_data, dict), type(
            instance.example_section_data
        )

        instance.example_section_data = Section.from_json(instance.example_section_data)

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
        # print(set(instance.content.keys()) - set(sections_))
        # for s in sections_:
        #    #for i, p in enumerate(instance.content[s]):
        #    #    if p[2]:
        #    #        instance.content[s][i] = (p[0], p[1], paragraphs(p[2]))

        ### dive into the example data, reconstruct the initial code, parse it with pygments,
        # and append the highlighting class as the third element
        # I'm thinking the linking strides should be stored separately as the code
        # it might be simpler, and more compact.
        # TODO : move this to ingest.
        instance.example_section_data = processed_example_data(
            instance.example_section_data, qa
        )

        for section in ["Extended Summary", "Summary", "Notes"] + sections_:
            if (data := instance.content.get(section, None)) is not None:
                assert isinstance(data, (list, dict)), f"{section} {data}"
                if data == []:
                    instance.content[section] = Section()
                else:
                    instance.content[section] = Section.from_json(data)

        for section in ["Extended Summary", "Summary", "Notes"] + sections_:
            if (data := instance.content.get(section, None)) is not None:
                assert isinstance(data, Section), data

        local_refs = []
        for s in sections_:
            from .take2 import Param

            local_refs = local_refs + [
                x[0] for x in instance.content[s] if isinstance(x, Param)
            ]
        visitor = DirectiveVisiter(qa, frozenset(), local_refs)
        for section in ["Extended Summary", "Summary", "Notes"] + sections_:
            assert section in instance.content
            instance.content[section] = visitor.visit(instance.content[section])
        if len(visitor.local) or len(visitor.total):
            print(f"{len(visitor.local)} / {len(visitor.total)}")

        return instance

    def to_json(self):

        res = {
            k: getattr(self, k, "")
            for k in self.slots()
            if k not in {"example_section_data", "see_also"}
        }
        res["example_section_data"] = self.example_section_data.to_json()
        res["see_also"] = [s.to_json() for s in self.see_also]

        return res


@lru_cache
def _at_in(q0, known_refs):
    return [q for q in known_refs if q.startswith(q0)]


def resolve_(qa: str, known_refs, local_ref, ref):
    if ref.startswith("builtins."):
        return ref, "missing"
    if ref.startswith("str."):
        return ref, "missing"
    if ref in {"None", "False", "True"}:
        return ref, "missing"
    # here is sphinx logic.
    # https://www.sphinx-doc.org/en/master/_modules/sphinx/domains/python.html?highlight=tilde
    # tilda ~ hide the module name/class name
    # dot . search more specific first.
    if ref.startswith("~"):
        ref = ref[1:]
    if ref in local_ref:
        return ref, "local"
    if ref in known_refs:
        return ref, "exists"
    else:
        if ref.startswith("."):
            if (found := qa + ref) in known_refs:
                return found, "exists"
            else:
                root = qa.split(".")[0]
                subset = [
                    r for r in known_refs if r.startswith(root) and r.endswith(ref)
                ]
                if len(subset) == 1:
                    return subset[0], "exists"
                else:
                    if len(subset) > 1:
                        # ambiguous ref
                        # print("subset", subset, ref, root)
                        pass

                # print(f"did not resolve {qa} + {ref}")
                return ref, "missing"

        parts = qa.split(".")
        for i in range(len(parts)):
            attempt = ".".join(parts[:i]) + "." + ref
            if attempt in known_refs:
                return attempt, "exists"

    q0 = qa.split(".")[0]
    attempts = [q for q in _at_in(q0, known_refs) if (ref in q)]
    if len(attempts) == 1:
        return attempts[0], "exists"
    return ref, "missing"


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if hasattr(o, "to_json"):
            return o.to_json()
        return super().default(o)


def assert_normalized(ref):
    # nref = normalise_ref(ref)
    # assert nref == ref, f"{nref} != {ref}"
    return ref


def load_one_uningested(bytes_, bytes2_, qa=None) -> IngestedBlobs:
    """
    Load the json from a DocBlob and make it an ingested blob.
    """
    data = json.loads(bytes_)

    instance = DocBlob.from_json(data)

    blob = IngestedBlobs()
    # TODO: here or maybe somewhere else:
    # see also 3rd item description is improperly deserialised as now it can be a paragraph.
    # Make see Also an auto deserialised object in take2.
    blob.see_also = [SeeAlsoItem.from_json(x) for x in data.pop("see_also", [])]

    for k in instance.slots():
        setattr(blob, k, getattr(instance, k))

    # blob = IngestedBlobs.from_json(data, rehydrate=False)
    # blob._parsed_data = data.pop("_parsed_data")
    data.pop("_parsed_data", None)
    data.pop("example_section_data", None)
    blob.refs = data.pop("refs", [])
    if bytes2_ is not None:
        backrefs = json.loads(bytes2_)
    else:
        backrefs = []
    blob.backrefs = backrefs
    blob.version = data.pop("version", "")

    if (see_also := blob.content.get("See Also", None)) and not blob.see_also:
        for nts, d0 in see_also:
            try:
                d = d0
                for (name, type_or_description) in nts:
                    if type_or_description and not d:
                        desc = type_or_description
                        if isinstance(desc, str):
                            desc = [desc]
                        assert isinstance(desc, list)
                        desc = paragraphs(desc)
                        type_ = None
                    else:
                        desc = d0
                        type_ = type_or_description
                        assert isinstance(desc, list)
                        desc = paragraphs(desc)

                    sai = SeeAlsoItem(Ref(name, None, None), desc, type_)
                    blob.see_also.append(sai)
                    del desc
                    del type_
            except Exception as e:
                raise ValueError(
                    f"Error {qa}: {see_also=}    |    {nts=}    | {d0=}"
                ) from e

    assert isinstance(blob.see_also, list), f"{blob.see_also=}"
    for l in blob.see_also:
        assert isinstance(l, SeeAlsoItem), f"{blob.see_also=}"
    blob.see_also = list(set(blob.see_also))

    # here we parse the example_section_data text paragraph into their
    # detailed representation of tokens this will simplify finding references
    # at ingestion time.
    # we also need to move this step at generation time,as we (likely), want to
    # do some local pre-processing of the references to already do some resolutions.
    from .gen import Section

    sec = Section.from_json(blob.example_section_data)
    new_sec = Section()

    for in_out in sec:
        type_ = in_out.__class__.__name__
        assert type_ in ("Code", "Text", "Fig"), f"{type_}, {in_out}"
        if type_ == "Text":
            pass
            # !!! MOVE This To GEN ?
            # assert isinstance(in_out, str), repr(in_out)
            # ps = paragraphs(in_out.split("\n"))
            # blob.example_section_data[i][1] = ps
            # for ps in in_out:
            #    for p in ps:
            #        assert p.__class__.__name__ in {
            #            "Word",
            #            "Verbatim",
            #            "Directive",
            #            "Math",
            #        }, f"{p[0]}, {qa}"
            blocks = P2(in_out.value.split("\n"))
            for b in blocks:
                new_sec.append(b)
        else:
            new_sec.append(in_out)

    blob.example_section_data = new_sec

    try:
        notes = blob.content["Notes"]
        if notes:
            blob.refs.extend(Paragraph.parse_lines(notes).references)
    except KeyError:
        pass
    for section in ["Extended Summary", "Summary", "Notes", "Warnings"]:
        if section in instance.content:
            if data := instance.content[section]:
                instance.content[section] = Section(P2(data))
            else:
                instance.content[section] = Section()

    blob.refs = list(sorted(set(blob.refs)))
    for section in ["Extended Summary", "Summary", "Notes", "Warnings"]:
        if (data := blob.content.get(section, None)) is not None:
            assert isinstance(data, Section), f"{data} {section}"

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
    from .take2 import Param

    for s in sections_:
        if s in blob.content:
            assert isinstance(blob.content[s], list)
            new_content = Section()
            for param, type_, desc in blob.content[s]:
                assert isinstance(desc, list)
                blocks = []
                items = []
                if desc:
                    items = P2(desc)
                new_content.append(Param(param, type_, items))
            blob.content[s] = new_content

    local_refs = []
    for s in sections_:

        local_refs = local_refs + [
            x[0] for x in instance.content[s] if isinstance(x, Param)
        ]
    visitor = DirectiveVisiter(qa, frozenset(), local_refs)
    for section in ["Extended Summary", "Summary", "Notes"] + sections_:
        assert section in instance.content
        instance.content[section] = visitor.visit(instance.content[section])

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
    def __init__(self, qa, known_refs, local_refs):
        assert isinstance(qa, str), qa
        assert isinstance(known_refs, (list, set, frozenset)), known_refs
        self.known_refs = known_refs
        self.local_refs = local_refs
        self.qa = qa
        self.local = []
        self.total = []

    def replace_Directive(self, directive):
        if (directive.domain is not None) or (directive.role not in (None, "mod")):
            return [directive]
        ref, exists = resolve_(
            self.qa, self.known_refs, self.local_refs, directive.text
        )
        if exists != "missing":
            if exists == "local":
                self.local.append(directive.text)
            else:
                self.total.append((directive.text, ref))
            return [Link(directive.text, ref, exists, exists != "missing")]
        return [directive]


def load_one(bytes_, bytes2_, qa=None) -> IngestedBlobs:
    data = json.loads(bytes_)
    blob = IngestedBlobs.from_json(data, qa=qa)

    # blob._parsed_data = data.pop("_parsed_data")
    data.pop("_parsed_data", None)
    data.pop("example_section_data", None)
    # blob._parsed_data["Parameters"] = [
    #    Parameter(a, b, c) for (a, b, c) in blob._parsed_data["Parameters"]
    # ]
    blob.backrefs = json.loads(bytes2_) if bytes2_ else []
    blob.version = data.pop("version", "")
    blob.refs = data.pop("refs", [])

    ## verification:

    # for i, (type_, (in_out)) in enumerate(blob.example_section_data):
    #        if type_ == "code":
    #            assert len(in_out) == 2
    #        if type_ == "text":
    #            for type_,word in in_out:
    #                assert isinstance(type_, str)
    #                assert isinstance(word, str)

    return blob


class Ingester:
    def __init__(self):
        self.ingest_dir = ingest_dir

    def ingest(self, path: Path, check: bool):
        nvisited_items = {}
        other_backrefs = {}
        root = None
        meta_path = path / "papyri.json"
        with meta_path.open() as f:
            data = json.loads(f.read())
            version = data["version"]
            logo = data.get("logo", None)
            aliases = data.get("aliases", {})
            root = data.get("module")

        del f

        for p, f1 in progress(
            (path / "module").glob("*.json"),
            description=f"Reading {path} doc bundle files ...",
        ):
            qa = f1.name[:-5]
            if check or True:
                rqa = normalise_ref(qa)
                if rqa != qa:
                    # numpy weird thing
                    print(f"skip {qa}")
                    continue
                assert rqa == qa, f"{rqa} !+ {qa}"
            try:
                with f1.open() as fff:
                    brpath = Path(str(f1)[:-5] + "br")
                    br: Optional[str]
                    if brpath.exists():
                        br = brpath.read_text()
                    else:
                        br = None
                    blob = load_one_uningested(fff.read(), br, qa=qa)
                    nvisited_items[qa] = blob
            except Exception as e:
                raise RuntimeError(f"error Reading to {f1}") from e
        del f1

        (self.ingest_dir / root).mkdir(exist_ok=True)
        (self.ingest_dir / root / version).mkdir(exist_ok=True)
        (self.ingest_dir / root / version / "module").mkdir(exist_ok=True)
        known_refs = frozenset(nvisited_items.keys())

        # TODO :in progress, crosslink needs version information.
        known_ref_info = frozenset(
            RefInfo(root, version, "api", qa) for qa in known_refs
        )

        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Cross referencing"
        ):
            local_ref = [x[0] for x in doc_blob.content["Parameters"] if x[0]] + [
                x[0] for x in doc_blob.content["Returns"] if x[0]
            ]
            doc_blob.logo = logo
            for ref in doc_blob.refs:
                resolved, exists = resolve_(qa, known_refs, local_ref, ref)
                # here need to check and load the new files touched.
                if resolved in nvisited_items and ref != qa:
                    nvisited_items[resolved].backrefs.append(qa)
                elif ref != qa and exists == "missing":
                    ref_root = ref.split(".")[0]
                    if ref_root == root:
                        continue
                    if ref_root == "builtins":
                        continue
                    from glob import escape as ge

                    existing_locations = list(
                        (self.ingest_dir / ref_root).glob(
                            f"*/module/{ge(resolved)}.json"
                        )
                    )
                    assert len(existing_locations) <= 1
                    if not existing_locations:
                        # print("Could not find", resolved, ref, f"({qa})")
                        continue
                    existing_location = existing_locations[0]
                    # existing_location = (
                    #    self.ingest_dir / ref_root / "module" / (resolved + ".json")
                    # )
                    if existing_location.exists():
                        brpath = Path(str(existing_location)[:-5] + ".br")
                        if brpath.exists():
                            br = brpath.read_text()
                        else:
                            br = None
                        try:
                            other_backrefs[resolved] = load_one(
                                existing_location.read_bytes(),
                                br,
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

            for sa in doc_blob.see_also:
                resolved, exists = resolve_(qa, known_refs, [], sa.name.name)
                if exists == "exists":
                    sa.name.exists = True
                    sa.name.ref = resolved
        (self.ingest_dir / root / version / "assets").mkdir(exist_ok=True)
        for px, f2 in progress(
            (path / "assets").glob("*"),
            description=f"Reading {path} image files ...",
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
            mod_root = qa.split(".")[0]
            assert mod_root == root
            # TODO : this is wrong, we get version of module we are currently ingesting
            doc_blob.version = version
            js = doc_blob.to_json()
            br = js.pop("backrefs", [])
            try:
                path = self.ingest_dir / mod_root / version / "module" / f"{qa}.json"
                path_br = self.ingest_dir / mod_root / version / "module" / f"{qa}.br"

                with path.open("w") as f:
                    f.write(json.dumps(js, cls=EnhancedJSONEncoder, indent=2))
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
                    f.write(json.dumps(js, cls=EnhancedJSONEncoder, indent=2))
                if path_br.exists():
                    bb = json.loads(path_br.read_text())
                else:
                    bb = []
                path_br.write_text(json.dumps(list(sorted(set(br + bb)))))
            except Exception as e:
                raise RuntimeError(f"error writing to {path}") from e


def main(path, check):
    print(path)
    Ingester().ingest(path, check)
