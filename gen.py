import numpy as np
import scipy
import scipy.special
import sklearn
from velin import parsedoc
from types import ModuleType

# import matplotlib
# import matplotlib.pyplot
import inspect

modules = [
    np,
    np.fft,
    np.ndarray,
    scipy,
    scipy.special,
    # sklearn,
    # matplotlib,
    # matplotlib.pyplot,
]

visited_items = {}

for mod in modules:
    if not mod.__name__.startswith(("num", "sci", "skl", "mat")):
        continue
    # print('exploring module', mod)
    for n in dir(mod):
        if n == "ufunc":
            continue
        a = getattr(mod, n)
        if isinstance(a, ModuleType):
            if a not in modules:
                modules.append(a)
            continue
        if getattr(a, "__module__", None) is None:
            continue
        if hasattr(a, "__qualname__"):
            qa = a.__module__ + "." + a.__qualname__
        else:
            qa = a.__module__ + "." + n
            # print('skipping', type(a), getattr(a, '__qualname__', None), f'({n}?)')
            # continue
        if not qa.startswith(("num", "sci", "skl", "mat")):
            continue
        if getattr(a, "__doc__", None) is None:
            # print('no doc for', a)
            continue
        sig = None
        try:
            sig = str(inspect.signature(a))
        except (ValueError, TypeError):
            pass
        try:
            doc, warnings = parsedoc(a.__doc__, name=qa, sig=sig)
        except:
            continue
        if warnings:
            print(qa)
            for w in warnings:
                print("  |", w)
        sa = doc.see_also()
        if getattr(visited_items, qa, None):
            raise ValueError(f"{qa} already visited")
        visited_items[qa] = doc
        # if sa:
        # print(qa)
        # print(sa)
        # with open(f'html/{qa}.html','w') as f:
        #    f.write(s)

print(visited_items.keys())


def _a(qa, backref):
    return backref


def _b(qa, backref):
    root, _ = qa.rsplit(".", 1)
    return root + "." + backref


def resolver(qa, visited_items, ref):
    for x in _a, _b:
        br = x(qa, ref)
        if br in visited_items:
            return br
    return None


for qa, doc in visited_items.items():
    sa = doc.see_also()
    if not sa:
        continue
    for backref in sa:
        for x in _a, _b:
            br = x(qa, backref)
            if br in visited_items:
                visited_items[br].backrefs.append(qa)
                print(br, "<-", qa)
                break
        else:
            print("???", qa, "-?>", backref)
            pass

for qa, doc in visited_items.items():
    s = doc._repr_html_(lambda ref: resolver(qa, visited_items, ref))
    with open(f"html/{qa}.html", "w") as f:
        f.write(s)
