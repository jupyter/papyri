import json
import os
from functools import lru_cache
from types import ModuleType

from jinja2 import (ChoiceLoader, Environment, FileSystemLoader, PackageLoader,
                    select_autoescape)
from numpydoc.docscrape import Parameter

from velin import NumpyDocString


@lru_cache()
def keepref(ref):
    """
    Just a filter to remove a bunch of frequent refs and not cluter the ref list in examples.
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

    Refs are sometime import path, not fully qualified names, tough type inference in examples regularly give us fully
    qualified names. when visiting a ref, this tries to import it and replace it by the normal fullqual form.

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
            print("object is instance and should not be documented ?", repr(obj))
            return ref

        nref = obj.__module__ + "." + obj.__name__
        return nref
    except Exception:
        print("couldnot normalize", ref)
        pass
    return ref


if __name__ == "__main__":

    nvisited_items = {}
    for fname in os.listdir("cache"):
        qa = fname[:-5]
        qa = normalise_ref(qa)
        with open("cache/" + fname) as f:
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

    def resolve(ref):
        return ref

    @lru_cache()
    def exists(ref):
        if ref in nvisited_items:
            return "exists"
        else:
            if not ref.startswith(("builtins.", "__main__")):
                print(ref, "missing")
            return "missing"

    env = Environment(
        loader=FileSystemLoader("velin"),
        autoescape=select_autoescape(["html", "xml", "tpl", "tpl.j2"]),
    )
    template = env.get_template("core.tpl.j2")
    env.globals["resolve"] = resolve
    env.globals["exists"] = exists

    for qa, ndoc in nvisited_items.items():
        for ref in ndoc.refs:
            # print("trying", qa, "<-", ref)
            if (ref) in nvisited_items and ref != qa:
                nvisited_items[ref].backrefs.append(qa)

    for qa, ndoc in nvisited_items.items():
        # ndoc = nvisited_items[qa]
        # sa = doc.see_also()
        nsa = ndoc.refs
        # if not sa:
        #    continue
        # for backref in sa:
        #    for x in _a, _b:
        #        br = x(qa, backref)
        #        if br in visited_items:
        #            visited_items[br].backrefs.append(qa)
        #            nvisited_items[br].backrefs.append(qa)
        #            #print(br, "<-", qa)
        #            break
        #    else:
        #        #print("???", qa, "-?>", backref)
        pass

    for qa, ndoc in nvisited_items.items():
        # s = doc._repr_html_(lambda ref: resolver(qa, visited_items, ref))
        # ndoc = nvisited_items[qa]

        with open(f"html/{qa}.html", "w") as f:
            f.write(template.render(doc=ndoc, qa=qa))
