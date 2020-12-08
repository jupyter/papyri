import dataclasses
import json
import warnings
from functools import lru_cache
from pathlib import Path


from .config import ingest_dir
from .gen import normalise_ref, DocBlob
from .take2 import Paragraph
from .utils import progress

from there import print
from .core import Ref, SeeAlsoItem

warnings.simplefilter("ignore", UserWarning)


from typing import Optional


class IngestedBlobs(DocBlob):

    __slots__ = ("backrefs", "see_also", "version", "logo")
    # see_also: List[SeeAlsoItem]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backrefs = []

    def slots(self):
        return super().slots() + ["backrefs", "see_also", "version", "logo"]


@lru_cache()
def keepref(ref):
    """
    Filter to rim out common reference that we usually do not want to keep
    around in examples; typically most of the builtins, and things we can't
    import.
    """
    if ref.startswith(("builtins.", "__main__")):
        return False
    try:
        __import__(ref)
        return False
    except Exception:
        pass
    return True


def resolve_(qa: str, known_refs, local_ref):
    def resolve(ref):
        if ref in local_ref:
            return ref, "local"
        if ref in known_refs:
            return ref, "exists"
        else:
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
        return super().default(o)


def assert_normalized(ref):
    # nref = normalise_ref(ref)
    # assert nref == ref, f"{nref} != {ref}"
    return ref


def load_one(bytes_, bytes2_, qa=None) -> IngestedBlobs:
    # blob.see_also = [SeeAlsoItem.from_json(**x) for x in data.pop("see_also", [])]

    data = json.loads(bytes_)
    blob = IngestedBlobs.from_json(data)
    # blob._parsed_data = data.pop("_parsed_data")
    data.pop("_parsed_data", None)
    data.pop("example_section_data", None)
    assert "edata" not in data
    # blob._parsed_data["Parameters"] = [
    #    Parameter(a, b, c) for (a, b, c) in blob._parsed_data["Parameters"]
    # ]
    blob.refs = data.pop("refs", [])
    # blob.edata = data.pop("edata")
    if bytes2_ is not None:
        backrefs = json.loads(bytes2_)
    else:
        backrefs = []
    blob.backrefs = backrefs
    # blob.see_also = data.pop("see_also", [])
    blob.see_also = [SeeAlsoItem.from_json(**x) for x in data.pop("see_also", [])]
    blob.version = data.pop("version", "")
    data.pop("ordered_sections", None)
    data.pop("backrefs", None)
    data.pop("_content", None)
    data.pop("item_file", None)
    data.pop("item_line", None)
    data.pop("item_type", None)
    data.pop("logo", None)
    if data.keys():
        print(data.keys(), qa)
        raise ValueError("remaining data", data.keys())
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
    blob.see_also = list(set(blob.see_also))
    try:
        notes = blob.content["Notes"]
        blob.refs.extend(Paragraph.parse_lines(notes).references)
    except KeyError:
        pass
    blob.refs = list(sorted(set(blob.refs)))
    return blob


class Ingester:
    def __init__(self):
        self.ingest_dir = ingest_dir

    def ingest(self, path: Path, check: bool):

        nvisited_items = {}
        versions = {}
        root = None
        for console, meta_path in progress(
            path.glob("**/__papyri__.json"),
            description="Loading package versions...",
        ):
            with meta_path.open() as f:
                data = json.loads(f.read())
                version = data["version"]
                logo = data.get("logo", None)
            versions[meta_path.parent.name] = version
            root = str(meta_path).split("/")[1]
        for p, f in progress(
            path.glob(f"{root}/*.json"),
            description=f"Reading {path} doc bundle files ...",
        ):
            if f.is_dir():
                continue
            fname = f.name
            if fname == "__papyri__.json":
                continue
            qa = fname[:-5]
            if check:
                rqa = normalise_ref(qa)
                if rqa != qa:
                    # numpy weird thing
                    print(f"skip {qa}")
                    continue
                assert rqa == qa, f"{rqa} !+ {qa}"
            try:
                with f.open() as fff:
                    from pathlib import Path

                    brpath = Path(str(f)[:-5] + "br")
                    br: Optional[str]
                    if brpath.exists():
                        br = brpath.read_text()
                    else:
                        br = None
                    blob = load_one(fff.read(), br, qa=qa)
                    if check:
                        blob.refs = [
                            assert_normalized(ref) for ref in blob.refs if keepref(ref)
                        ]
                    nvisited_items[qa] = blob

            except Exception as e:
                raise RuntimeError(f"error Reading to {f}") from e

        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Cross referencing"
        ):
            local_ref = [x[0] for x in doc_blob.content["Parameters"] if x[0]] + [
                x[0] for x in doc_blob.content["Returns"] if x[0]
            ]
            doc_blob.logo = logo
            for ref in doc_blob.refs:
                resolved, exists = resolve_(qa, nvisited_items, local_ref)(ref)
                # here need to check and load the new files touched.
                if resolved in nvisited_items and ref != qa:
                    # print(qa, 'incommon')
                    nvisited_items[resolved].backrefs.append(qa)
                elif ref != qa and exists == "missing":
                    ex = self.ingest_dir / (resolved + ".json")
                    if ex.exists():
                        with ex.open() as f:
                            brpath = Path(str(ex)[:-5] + "br")
                            if brpath.exists():
                                br = brpath.read_text()
                            else:
                                br = None
                            blob = load_one(f.read(), br)
                            nvisited_items[resolved] = blob
                            if not hasattr(nvisited_items[resolved], "backrefs"):
                                nvisited_items[resolved].backrefs = []
                            nvisited_items[resolved].backrefs.append(qa)
                    elif "/" not in resolved:
                        phantom_dir = self.ingest_dir / "__phantom__"
                        phantom_dir.mkdir(exist_ok=True)
                        ph = phantom_dir / (resolved + ".json")
                        if ph.exists():
                            with ph.open() as f:
                                ph_data = json.loads(f.read())

                        else:
                            ph_data = []
                        ph_data.append(qa)
                        with ph.open("w") as f:
                            # print('updating phantom data', ph)
                            f.write(json.dumps(ph_data))
                    else:
                        print(resolved, "not valid reference, skipping.")

            for sa in doc_blob.see_also:
                resolved, exists = resolve_(qa, nvisited_items, [])(sa.name.name)
                if exists == "exists":
                    sa.name.exists = True
                    sa.name.ref = resolved
        for px, f in progress(
            path.glob(f"**/*.png"),
            description=f"Reading {path} image files ...",
        ):
            with open(self.ingest_dir / f.name, "wb") as fw:
                fw.write(f.read_bytes())

        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Cleaning double references"
        ):
            # TODO: load backrref from either:
            # 1) previsous version fo the file
            # 2) phantom file if first import (and remove the phantom file?)
            phantom_dir = self.ingest_dir / "__phantom__"
            ph = phantom_dir / (qa + ".json")
            # print('ph?', ph)
            if ph.exists():
                with ph.open() as f:
                    ph_data = json.loads(f.read())
                print("loading data from phantom file !", ph_data)
            else:
                ph_data = []

            doc_blob.backrefs = list(sorted(set(doc_blob.backrefs + ph_data)))
        for console, path in progress(
            self.ingest_dir.glob("{root}/*.json"),
            description="cleanig previsous files ....",
        ):
            path.unlink()

        for p, (qa, doc_blob) in progress(
            nvisited_items.items(), description="Writing..."
        ):
            root = qa.split(".")[0]
            doc_blob.version = versions.get(root, "?")
            js = doc_blob.to_json()
            br = js.pop("backrefs", [])
            try:
                path = self.ingest_dir / f"{qa}.json"
                path_br = self.ingest_dir / f"{qa}.br"

                with path.open("w") as f:
                    f.write(json.dumps(js, cls=EnhancedJSONEncoder))
                if path_br.exists():
                    with path_br.open("r") as f:
                        bb = json.loads(f.read())
                else:
                    bb = []
                with path_br.open("w") as f:
                    f.write(json.dumps(list(sorted(set(br + bb)))))
            except Exception as e:
                raise RuntimeError(f"error writing to {fname}") from e


def main(path, check):
    Ingester().ingest(path, check)
