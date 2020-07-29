import dataclasses
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

from velin import NumpyDocString

from numpydoc.docscrape import Parameter

from .config import base_dir, cache_dir, ingest_dir
from .gen import keepref, normalise_ref
from .take2 import Paragraph
from .utils import progress


def resolve_(qa, known_refs, local_ref):
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


@dataclass
class Ref:
    name: str
    ref: str
    exists: bool


@dataclass
class SeeAlsoItem:
    name: Ref
    descriptions: str
    # there are a few case when the lhs is `:func:something`... in scipy.
    type: str

    @classmethod
    def from_json(cls, name, descriptions, type):
        return cls(Ref(**name), descriptions, type)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def assert_normalized(ref):
    # nref = normalise_ref(ref)
    # assert nref == ref, f"{nref} != {ref}"
    return ref


def main(check):

    nvisited_items = {}
    for p, f in progress(cache_dir.glob("**/*.json"), description="Loading..."):
        if f.is_dir():
            continue
        fname = f.name
        qa = fname[:-5]
        if check:
            rqa = normalise_ref(qa)
            assert rqa == qa, f"{rqa} !+ {qa}"
        try:
            with f.open() as f:
                data = json.loads(f.read())
                blob = NumpyDocString("")
                blob._parsed_data = data["_parsed_data"]
                blob._parsed_data["Parameters"] = [
                    Parameter(a, b, c) for (a, b, c) in blob._parsed_data["Parameters"]
                ]
                if check:
                    test = assert_normalized
                    keep = keepref
                else:
                    test = lambda x: x
                    keep = lambda x: True
                blob.refs = [test(ref) for ref in data["refs"] if keep(ref)]
                blob.edata = data["edata"]
                blob.backrefs = data["backref"]
                blob.see_also = []
                try:
                    if see_also := blob["See Also"]:
                        for nts, d in see_also:
                            for (n, t) in nts:
                                if t and not d:
                                    d, t = t, None
                                blob.see_also.append(
                                    SeeAlsoItem(Ref(n, None, None), d, t)
                                )
                except Exception as e:
                    raise ValueError(f"Error {qa}: {see_also} | {nts}") from e

                nvisited_items[qa] = blob
                try:
                    notes = blob["Notes"]
                    blob.refs.extend(refs := Paragraph.parse_lines(notes).references)
                    # if refs:
                    # print(qa, refs)
                except KeyError:
                    pass
                blob.refs = list(sorted(set(blob.refs)))

        except Exception as e:
            raise RuntimeError(f"error Reading to {f}") from e

    # update teh backref ar render time. technically this shoudl be at
    # generation time, or even a separate step that can be optimized later, by
    # doing careful graph update of only referenced nodes.
    for p, (qa, ndoc) in progress(
        nvisited_items.items(), description="Cross referencing"
    ):
        local_ref = [x[0] for x in ndoc["Parameters"] if x[0]]
        for ref in ndoc.refs:
            resolved = resolve_(qa, nvisited_items, local_ref)(ref)[0]
            if resolved in nvisited_items and ref != qa:
                nvisited_items[resolved].backrefs.append(qa)

        for sa in ndoc.see_also:
            resolved, exists = resolve_(qa, nvisited_items, [])(sa.name.name)
            if exists == "exists":
                sa.name.exists = True
                sa.name.ref = resolved
    for console, path in progress(
        ingest_dir.glob("**/*.json"), description="cleanig old files...."
    ):
        path.unlink()
    for p, (qa, ndoc) in progress(nvisited_items.items(), description="Writing..."):
        ndoc.backrefs = list(sorted(set(ndoc.backrefs)))
        js = ndoc.to_json()
        try:
            path = ingest_dir / f"{qa}.json"
            with path.open("w") as f:
                f.write(json.dumps(js, cls=EnhancedJSONEncoder))
        except Exception as e:
            raise RuntimeError(f"error writing to {fname}") from e
