import dataclasses
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

from velin import NumpyDocString

from numpydoc.docscrape import Parameter

from .config import cache_dir
# from .render import resolve_
from .take2 import Paragraph
from .utils import progress


def resolve_(qa, known_refs):
    def resolve(ref):
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


@lru_cache()
def keepref(ref):
    """
    Just a filter to remove a bunch of frequent refs and not clutter the ref list in examples.
    """
    if ref.startswith(("builtins.", "__main__")):
        return False
    try:
        __import__(ref)
        return False
    except Exception:
        pass
    return True


@lru_cache()
def normalise_ref(ref):
    """
    Consistently normalize references.

    Refs are sometime import path, not fully qualified names, tough type
    inference in examples regularly give us fully qualified names. When visiting
    a ref, this tries to import it and replace it by the normal full-qualified form.

    """
    if ref.startswith(("builtins.", "__main__")):
        return ref
    try:
        mod_name, name = ref.rsplit(".", maxsplit=1)
        mod = __import__(mod_name)
        for sub in mod_name.split(".")[1:]:
            mod = getattr(mod, sub)
        obj = getattr(mod, name)
        if isinstance(obj, ModuleType):
            # print('module type.. skipping', ref)
            return ref

        if (
            getattr(obj, "__name__", None) is None
        ):  # and obj.__doc__ == type(obj).__doc__:
            # print("object is instance and should not be documented ?", repr(obj))
            return ref

        nref = obj.__module__ + "." + obj.__name__

        return nref
    except Exception:
        # print("could not normalize", ref)
        pass
    return ref




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


def main():

    nvisited_items = {}
    for p, f in progress(cache_dir.glob("*"), description="Loading..."):
        fname = f.name
        qa = fname[:-5]
        qa = normalise_ref(qa)
        try:
            with f.open() as f:
                data = json.loads(f.read())
                blob = NumpyDocString("")
                blob._parsed_data = data["_parsed_data"]
                blob._parsed_data["Parameters"] = [
                    Parameter(a, b, c) for (a, b, c) in blob._parsed_data["Parameters"]
                ]
                blob.refs = [normalise_ref(ref) for ref in data["refs"] if keepref(ref)]
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
        for ref in ndoc.refs:
            resolved = resolve_(qa, nvisited_items)(ref)[0]
            if resolved in nvisited_items and ref != qa:
                nvisited_items[resolved].backrefs.append(qa)

        for sa in ndoc.see_also:
            resolved, exists = resolve_(qa, nvisited_items)(sa.name.name)
            if exists == "exists":
                sa.name.exists = True
                sa.name.ref = resolved

    for p, (qa, ndoc) in progress(nvisited_items.items(), description="Writing..."):
        ndoc.backrefs = list(sorted(set(ndoc.backrefs)))
        js = ndoc.to_json()
        try:
            path = cache_dir / f"{qa}.json"
            with path.open("w") as f:
                f.write(json.dumps(js, cls=EnhancedJSONEncoder))
        except Exception as e:
            raise RuntimeError(f"error writing to {fname}") from e
