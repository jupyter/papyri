import json
import os
from functools import lru_cache
from types import ModuleType

from numpydoc.docscrape import Parameter

from render import resolve_
from take2 import Paragraph
from velin import NumpyDocString


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


if __name__ == "__main__":

    nvisited_items = {}
    for fname in os.listdir("cache"):
        qa = fname[:-5]
        qa = normalise_ref(qa)
        try:
            with open(fname := "cache/" + fname) as f:
                data = json.loads(f.read())
                blob = NumpyDocString("")
                blob._parsed_data = data["_parsed_data"]
                blob._parsed_data["Parameters"] = [
                    Parameter(a, b, c) for (a, b, c) in blob._parsed_data["Parameters"]
                ]
                blob.refs = [normalise_ref(ref) for ref in data["refs"] if keepref(ref)]
                blob.edata = data["edata"]
                blob.backrefs = data["backref"]
                nvisited_items[qa] = blob
                try:
                    notes = blob["Notes"]
                    blob.refs.extend(refs := Paragraph.parse_lines(notes).references)
                    if refs:
                        print(qa, refs)
                except KeyError:
                    pass
                blob.refs = list(sorted(set(blob.refs)))

        except Exception as e:
            raise RuntimeError(f"error writing to {fname}") from e

    # update teh backref ar render time. technically this shoudl be at
    # generation time, or even a separate step that can be optimized later, by
    # doing careful graph update of only referenced nodes.
    for qa, ndoc in nvisited_items.items():
        for ref in ndoc.refs:
            resolved = resolve_(qa, nvisited_items)(ref)[0]
            if resolved in nvisited_items and ref != qa:
                nvisited_items[resolved].backrefs.append(qa)

    for qa, ndoc in nvisited_items.items():
        ndoc.backrefs = list(sorted(set(ndoc.backrefs)))
        js = ndoc.to_json()
        try:
            fname = f"cache/{qa}.json"
            with open(fname, "w") as f:
                f.write(json.dumps(js))
        except Exception as e:
            raise RuntimeError(f"error writing to {fname}") from e
