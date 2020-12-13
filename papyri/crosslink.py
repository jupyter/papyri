import dataclasses
import json
import warnings
from functools import lru_cache
from pathlib import Path


from .config import ingest_dir
from .gen import normalise_ref, DocBlob
from .take2 import Lines, Paragraph, make_block_3, Math
from .utils import progress

from there import print
from .core import Ref, SeeAlsoItem

warnings.simplefilter("ignore", UserWarning)


from typing import Optional, List, Tuple, Any


def paragraph(lines) -> List[Tuple[str, str]]:
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


def paragraphs(lines) -> List[Any]:
    blocks = make_block_3(Lines(lines))
    acc = []
    for b0, b1, b2 in blocks:
        if b0:
            acc.append(paragraph([x._line for x in b0]))
        ## definitively wrong but will do for now, should likely be verbatim, or recurse ?
        if b2:
            acc.append(paragraph([x._line for x in b2]))
    return acc


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
    def from_json(cls, data):
        instance = super().from_json(data)
        instance.see_also = [
            SeeAlsoItem.from_json(**x) for x in data.pop("see_also", [])
        ]
        return instance

    def to_json(self):
        data = super().to_json()
        return data


def resolve_(qa: str, known_refs, local_ref):
    def resolve(ref):
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

            parts = qa.split(".")
            for i in range(len(parts)):
                attempt = ".".join(parts[:i]) + "." + ref
                if attempt in known_refs:
                    return attempt, "exists"

        q0 = qa.split(".")[0]
        attempts = [q for q in known_refs if q.startswith(q0) and (ref in q)]
        if len(attempts) == 1:
            return attempts[0], "exists"
        return ref, "missing"

    return resolve


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
    blob = IngestedBlobs.from_json(data)
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
    for i, (type_, (in_out)) in enumerate(blob.example_section_data):
        if type_ == "text":
            blob.example_section_data[i][1] = paragraphs([in_out])
            for ps in blob.example_section_data[i][1]:
                for p in ps:
                    assert p[0] in {
                        "Word",
                        "Verbatim",
                        "Directive",
                        "Math",
                    }, f"{p[0]}, {qa}"

    blob.see_also = list(set(blob.see_also))
    try:
        notes = blob.content["Notes"]
        blob.refs.extend(Paragraph.parse_lines(notes).references)
    except KeyError:
        pass
    blob.refs = list(sorted(set(blob.refs)))
    return blob


def load_one(bytes_, bytes2_, qa=None) -> IngestedBlobs:
    data = json.loads(bytes_)
    blob = IngestedBlobs.from_json(data)
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
        versions = {}
        root = None
        meta_path = path / "papyri.json"
        print(f"{path=}")
        with meta_path.open() as f:
            data = json.loads(f.read())
            version = data["version"]
            logo = data.get("logo", None)
            aliases = data.get("aliases", {})
            root = data.get("module")

        for p, f in progress(
            (path / "module").glob("*.json"),
            description=f"Reading {path} doc bundle files ...",
        ):
            qa = f.name[:-5]
            if check or True:
                rqa = normalise_ref(qa)
                if rqa != qa:
                    # numpy weird thing
                    print(f"skip {qa}")
                    continue
                assert rqa == qa, f"{rqa} !+ {qa}"
            try:
                with f.open() as fff:
                    brpath = Path(str(f)[:-5] + "br")
                    br: Optional[str]
                    if brpath.exists():
                        br = brpath.read_text()
                    else:
                        br = None
                    blob = load_one_uningested(fff.read(), br, qa=qa)
                    nvisited_items[qa] = blob
            except Exception as e:
                raise RuntimeError(f"error Reading to {f}") from e

        (self.ingest_dir / root).mkdir(exist_ok=True)
        (self.ingest_dir / root / "module").mkdir(exist_ok=True)

        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Cross referencing"
        ):
            local_ref = [x[0] for x in doc_blob.content["Parameters"] if x[0]] + [
                x[0] for x in doc_blob.content["Returns"] if x[0]
            ]
            doc_blob.logo = logo
            for ref in doc_blob.refs:
                resolved, exists = resolve_(qa, nvisited_items, local_ref)(ref)
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
                        with existing_location.open() as f:
                            brpath = Path(str(existing_location)[:-5] + ".br")
                            if brpath.exists():
                                br = brpath.read_text()
                            else:
                                br = None
                            nvisited_items[resolved] = load_one(f.read(), br)
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
                resolved, exists = resolve_(qa, nvisited_items, [])(sa.name.name)
                if exists == "exists":
                    sa.name.exists = True
                    sa.name.ref = resolved

        (self.ingest_dir / root / "assets").mkdir(exist_ok=True)
        for px, f in progress(
            (path / "assets").glob("*"),
            description=f"Reading {path} image files ...",
        ):
            (self.ingest_dir / root / "assets" / f.name).write_bytes(f.read_bytes())

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
                print("loading data from phantom file !", ph_data)
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
