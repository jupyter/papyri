import json
import os
from functools import lru_cache
from types import ModuleType

from jinja2 import (Environment, FileSystemLoader, PackageLoader,
                    select_autoescape)
from numpydoc.docscrape import Parameter

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
            print("object is instance and should not be documented ?", repr(obj))
            return ref

        nref = obj.__module__ + "." + obj.__name__
        return nref
    except Exception:
        print("could not normalize", ref)
        pass
    return ref


if __name__ == "__main__":

    nvisited_items = {}
    for fname in os.listdir("cache"):
        qa = fname[:-5]
        with open("cache/" + fname) as f:
            data = json.loads(f.read())
            blob = NumpyDocString("")
            blob._parsed_data = data["_parsed_data"]
            blob._parsed_data["Parameters"] = [
                Parameter(a, b, c) for (a, b, c) in blob._parsed_data["Parameters"]
            ]
            blob.refs = data["refs"]
            blob.edata = data["edata"]
            blob.backrefs = data["backref"]
            nvisited_items[qa] = blob

    # nothign for now, hardcoded to qualname + html, but shoudl be custom for
    # various renderer, spyder for example want likely a custom actions in their
    # inspector.
    def resolve(ref):
        return ref

    # TODO, make that a non-closure ?
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
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    template = env.get_template("core.tpl.j2")
    env.globals["exists"] = exists

    for qa, ndoc in nvisited_items.items():
        with open(f"html/{qa}.html", "w") as f:

            def resolve(ref):
                if ref in nvisited_items:
                    return ref, "exists"
                else:
                    parts = qa.split(".")
                    for i in range(len(parts)):
                        attempt = ".".join(parts[:i]) + "." + ref
                        if attempt in nvisited_items:
                            return attempt, "exists"
                return ref, "missing"

            env.globals["resolve"] = resolve

            br = ndoc.backrefs
            if len(br):
                print(len(br))
            if len(br) > 30:
                from collections import defaultdict

                b2 = defaultdict(lambda: [])
                for ref in br:
                    mod, _ = ref.split(".", maxsplit=1)
                    b2[mod].append(ref)
                backrefs = (None, b2)
            else:
                backrefs = (br, None)

            f.write(
                template.render(
                    doc=ndoc,
                    qa=qa,
                    version="X.y.z",
                    module=qa.split(".")[0],
                    backrefs=backrefs,
                )
            )
