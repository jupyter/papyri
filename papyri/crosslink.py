import dataclasses
import json
import warnings
from functools import lru_cache
from pathlib import Path


from .config import ingest_dir
from .gen import normalise_ref, DocBlob
from .take2 import Lines, Paragraph, make_block_3, Math, main as t2main, Line, Link
from .utils import progress

from there import print
from .core import Ref, SeeAlsoItem

warnings.simplefilter("ignore", UserWarning)


from typing import Optional, List, Tuple, Any, Dict


def paragraph(lines) -> List[Tuple[str, Any]]:
    """
    return container of (type, obj)
    """
    p = Paragraph.parse_lines(lines)
    acc = []
    for c in p.children:
        if type(c).__name__ == "Directive":
            if c.role == "math":
                acc.append(("Math", Math(c.value)))
            else:
                acc.append((type(c).__name__, c))
        else:
            acc.append((type(c).__name__, c))
    return acc


def P2(lines) -> List[Any]:
    assert isinstance(lines, list)
    for l in lines:
        if isinstance(l, str):
            assert "\n" not in l
        else:
            assert "\n" not in l._line
    acc = []
    assert lines
    blocks_data = t2main("\n".join(lines))

    # for pre_blank_lines, blank_lines, post_black_lines in blocks_data:
    # for block in blocks_data:
    #    print(block)
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


def processed_example_data(example_section_data, local_refs, qa):
    new_example_section_data = []
    for i, (type_, in_out) in enumerate(example_section_data):
        if type_ == "code":
            assert len(in_out) == 3

            in_, out, ce_status = in_out
            if len(in_[0]) == 2:
                classes = get_classes("".join([x for x, y in in_]))
                for ii, cc in zip(in_, classes):
                    # TODO: Warning here we mutate objects.
                    ii.append(cc)
        if type_ == "text":
            assert len(in_out) == 1, len(in_out)
            new_io = []
            for t_, it in in_out[0]:
                if it.__class__.__name__ == "Directive" and it.domain is None:
                    if it.domain is None and it.role is None:
                        ref, exists = resolve_(qa, frozenset(), local_refs, it.text)
                        if exists != "missing":
                            t_ = "Link"
                            it = Link(it.text, ref, exists, exists != "missing")
                    else:
                        print(f"unhandled {it.domain=}, {it.role=}, {it.text}")
                new_io.append((t_, it))
            in_out = [new_io]
        new_example_section_data.append((type_, in_out))
    return new_example_section_data


from pygments import lex
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


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
    def from_json(cls, data, rehydrate=True, qa=None):
        instance = super().from_json(data)
        instance.see_also = [
            SeeAlsoItem.from_json(**x) for x in data.pop("see_also", [])
        ]
        for d in instance.see_also:
            assert isinstance(d.descriptions, list), qa
        # Todo: remove this; hopefully the logic from load_one_uningested
        # load a DocBlob instaead of un IngestedDocBlob
        # or more likely the paragraph parsing is made in Gen.
        if rehydrate:
            assert qa is not None
            # rehydrate in the example section paragraphs into their actual
            # instances. This likely should be moved into a specific type of
            # instance in the Parsed Data.
            for i, (type_, (in_out)) in enumerate(instance.example_section_data):
                if type_ == "text":
                    from . import take2 as take2

                    assert isinstance(in_out, list), repr(in_out)
                    # assert len(in_out) == 1, f"{len(in_out)}"
                    new = []
                    for tt, value in in_out[0]:
                        assert tt in {
                            "Word",
                            "Verbatim",
                            "Directive",
                            "Math",
                            "Link",
                        }, f"{tt}, {value}"
                        constr = getattr(take2, tt)
                        nval = constr.from_json(value)
                        new.append((tt, nval))

                    # in_out is a paragraph.
                    instance.example_section_data[i][1] = [new]

            sections_ = [
                "Parameters",
                "Returns",
                "Raises",
                "Yields",
                "Attributes",
                "Other Parameters",
            ]
            for s in sections_:
                for i, p in enumerate(instance.content[s]):
                    if p[2]:
                        instance.content[s][i] = (p[0], p[1], paragraphs(p[2]))

            ### dive into the example data, reconstruct the initial code, parse it with pygments,
            # and append the highlighting class as the third element
            # I'm thinking the linking strides should be stored separately as the code
            # it might be simpler, and more compact.
            # TODO : move this to ingest.
            assert qa is not None
            local_refs = []
            for s in sections_:
                local_refs = local_refs + [x[0] for x in instance.content[s] if x[0]]
            instance.example_section_data = processed_example_data(
                instance.example_section_data, local_refs, qa
            )

        return instance

    def to_json(self):
        for i, (type_, in_out) in enumerate(self.example_section_data):
            if type_ == "text":
                assert isinstance(in_out, list), repr(in_out)
        data = super().to_json()
        return data

@lru_cache
def _at_in(q0, known_refs):
    return [q for q in known_refs if q.startswith(q0)]


def resolve_(qa: str, known_refs, local_ref, ref):
    assert isinstance(ref, str), type(ref)
    if ref.startswith("builtins."):
        return ref, "missing"
    if ref.startswith("str."):
        return ref, "missing"
    if ref in {"None", "False", "True"}:
        return ref, "missing"

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
    blob = IngestedBlobs.from_json(data, rehydrate=False)
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

    try:
        if (see_also := blob.content.get("See Also", None)) and not blob.see_also:
            for nts, d in see_also:
                for (n, t) in nts:
                    if t and not d:
                        d, t = t, None
                    if not isinstance(d, list):
                        d = [d]
                    blob.see_also.append(SeeAlsoItem(Ref(n, None, None), d, t))
    except Exception as e:
        raise ValueError(f"Error {qa}: {see_also} | {nts}") from e

    assert isinstance(blob.see_also, list), f"{blob.see_also=}"
    for l in blob.see_also:
        assert isinstance(l, SeeAlsoItem), f"{blob.see_also=}"

    # here we parse the example_section_data text paragraph into their
    # detailed representation of tokens this will simplify finding references
    # at ingestion time.
    # we also need to move this step at generation time,as we (likely), want to
    # do some local pre-processing of the references to already do some resolutions.
    for i, (type_, in_out) in enumerate(blob.example_section_data):
        if type_ == "text":
            assert isinstance(in_out, str), repr(in_out)
            ps = paragraphs(in_out.split("\n"))
            blob.example_section_data[i][1] = ps
            for ps in blob.example_section_data[i][1]:
                for p in ps:
                    assert p[0] in {
                        "Word",
                        "Verbatim",
                        "Directive",
                        "Math",
                    }, f"{p[0]}, {qa}"

    for i, (type_, in_out) in enumerate(blob.example_section_data):
        if type_ == "text":
            assert isinstance(in_out, list), repr(in_out)
            # assert len(in_out) == 1, f"{repr(in_out)}"

    blob.see_also = list(set(blob.see_also))
    try:
        notes = blob.content["Notes"]
        if notes:
            blob.refs.extend(Paragraph.parse_lines(notes).references)
    except KeyError:
        pass
    blob.refs = list(sorted(set(blob.refs)))
    return blob


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
        versions: Dict[Any, Any] = {}
        root = None
        meta_path = path / "papyri.json"
        print(f"{path=}")
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
        (self.ingest_dir / root / "module").mkdir(exist_ok=True)
        known_refs = frozenset(nvisited_items.keys())
    
        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Cross referencing"
        ):
            local_ref = [x[0] for x in doc_blob.content["Parameters"] if x[0]] + [
                x[0] for x in doc_blob.content["Returns"] if x[0]
            ]
            doc_blob.logo = logo
            for ref in doc_blob.refs:
                resolved, exists = resolve_(qa, known_refs, local_ref, ref)
                pp = False
                if "Audio" in qa:
                    pp = True
                    print("ref to", ref)
                # here need to check and load the new files touched.
                if resolved in nvisited_items and ref != qa:
                    nvisited_items[resolved].backrefs.append(qa)
                elif ref != qa and exists == "missing":
                    ref_root = ref.split(".")[0]
                    if ref_root == root:
                        continue
                    existing_location = (
                        self.ingest_dir / ref_root / "module" / (resolved + ".json")
                    )
                    if existing_location.exists():
                        brpath = Path(str(existing_location)[:-5] + ".br")
                        if brpath.exists():
                            br = brpath.read_text()
                        else:
                            br = None
                        nvisited_items[resolved] = load_one(
                            existing_location.read_bytes(), br, qa=resolved
                        )
                        nvisited_items[resolved].backrefs.append(qa)
                    elif "/" not in resolved:
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
        (self.ingest_dir / root / "assets").mkdir(exist_ok=True)
        for px, f2 in progress(
            (path / "assets").glob("*"), description=f"Reading {path} image files ...",
        ):
            (self.ingest_dir / root / "assets" / f2.name).write_bytes(f2.read_bytes())

        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Cleaning double references"
        ):
            # TODO: load backrref from either:
            # 1) previsous version fo the file
            # 2) phantom file if first import (and remove the phantom file?)
            phantom_dir = self.ingest_dir / root / "module" / "__phantom__"
            ph = phantom_dir / (qa + ".json")
            if ph.exists():
                ph_data = json.loads(ph.read_text())
            else:
                ph_data = []

            doc_blob.backrefs = list(sorted(set(doc_blob.backrefs + ph_data)))
        for console, path in progress(
            (self.ingest_dir / root / "module").glob("*.json"),
            description="cleanig previsous files ....",
        ):
            path.unlink()

        with open(self.ingest_dir / root / "papyri.json", "w") as f:
            f.write(json.dumps(aliases))


        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Writing..."
        ):
            # we might update other modules with backrefs
            mod_root = qa.split(".")[0]
            doc_blob.version = version
            js = doc_blob.to_json()
            br = js.pop("backrefs", [])
            try:
                path = self.ingest_dir / mod_root / "module" / f"{qa}.json"
                path_br = self.ingest_dir / mod_root / "module" / f"{qa}.br"

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
