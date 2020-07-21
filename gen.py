import inspect
from textwrap import dedent

# from numpydoc.docscrape import NumpyDocString
from types import ModuleType

import jedi
import matplotlib
import matplotlib.pyplot
import numpy as np
import numpy.core.numeric
import scipy
import scipy.special
import sklearn
from pygments.lexers import PythonLexer
from there import print

from velin import NumpyDocString, parsedoc
from velin.examples_section_utils import InOut, splitblank, splitcode
from velin.ref import NumpyDocString


def dedent_but_first(text):
    a, *b = text.split("\n")
    return dedent(a) + "\n" + dedent("\n".join(b))


def pos_to_nl(script, pos):
    rest = pos
    ln = 0
    for line in script.splitlines():
        if len(line) < rest:
            rest -= len(line) + 1
            ln += 1
        else:
            return ln, rest


P = PythonLexer()


def parse_script(script, ns=None):

    jeds = []
    if ns:
        jeds.append(jedi.Interpreter(script, namespaces=[ns]))
    jeds.append(jedi.Script(script))

    for index, type_, text in P.get_tokens_unprocessed(script):
        a, b = pos_to_nl(script, index)
        try:
            ref = None
            for jed in jeds:
                try:
                    ref = jed.infer(a + 1, b)[0].full_name
                except (AttributeError, TypeError, Exception):
                    pass
                break
        except IndexError:
            ref = ""
        yield index, type_, text, ref


def get_example_data(doc):
    blocks = list(map(splitcode, splitblank(doc["Examples"])))
    edata = []
    for b in blocks:
        for item in b:
            if isinstance(item, InOut):
                script = "\n".join(item.in_)
                entries = list(parse_script(script, ns={"np": np}))
                edata.append(["code", (entries, "\n".join(item.out))])

            else:
                edata.append(["text", "\n".join(item.out)])
    return edata


def main():

    import sys

    [do_one_mod(x) for x in sys.argv[1:]]


def do_one_mod(name):
    modules = [__import__(name)]
    root = name.split(".")[0]
    nvisited_items = {}
    for mod in modules:
        if not mod.__name__.startswith(root):
            print("\nskip", mod)
            # continue
            pass
        # print('exploring module', mod)
        for n in dir(mod):
            if n == "ufunc":
                continue
            try:
                a = getattr(mod, n)
            except Exception:
                continue
            if isinstance(a, ModuleType):
                if a not in modules:
                    pass
                    modules.append(a)
                continue
            if getattr(a, "__module__", None) is None:
                continue
            if isinstance(lqa := getattr(a, "__qualname__", None), str):
                qa = a.__module__ + "." + lqa
            else:
                qa = a.__module__ + "." + n
                # print('skipping', type(a), getattr(a, '__qualname__', None), f'({n}?)')
                # continue
            if not qa.startswith(root):
                # print('\nwrong mod for ', qa, repr(a)[:20]+'...', 'while visiting', name)
                continue
            if not isinstance(ddd := getattr(a, "__doc__", None), str):
                # print('no doc for', a)
                continue
            # sig = None
            # try:
            #    sig = str(inspect.signature(a))
            # except (ValueError, TypeError):
            #    pass
            try:
                # doc, warnings = parsedoc(a.__doc__, name=qa, sig=sig)
                ndoc = NumpyDocString(dedent_but_first(ddd))
            except:
                print("\nfailed", a)
                continue
            # if warnings:
            #    print(qa)
            #    for w in warnings:
            #        print("  |", w)
            nsa = ndoc["See Also"]
            refs = []
            if nsa:
                for line in nsa:
                    rt, desc = line
                    for ref, type_ in rt:
                        # ref, type_ = rt
                        refs.append(ref)
                        # ?assert type_ is None
                        # print( '   >', ref)
                    # print( '   >    ', desc)
            # sa = doc.see_also()
            # left, right = set(sa)-set(refs), set(refs)-set(sa)
            # if left or right:
            #    print('   :', left, right)
            #    print('   :', nsa)
            #    print('   :', a.__doc__)
            if getattr(nvisited_items, qa, None):
                raise ValueError(f"{qa} already visited")
            # visited_items[qa] = doc
            ndoc.backrefs = []
            ndoc.edata = get_example_data(ndoc)
            # if 'Examples' in a.__doc__:
            #    print(a, qa, 'has examples')
            # if ndoc['Examples']:
            #    print(ndoc.edata, qa)
            ndoc.refs = list(
                {
                    u[3]
                    for t_, sect in ndoc.edata
                    if t_ == "code"
                    for u in sect[0]
                    if u[3]
                }
            )
            ndoc.backrefs = []
            # print(' '+qa+' '*30)
            import json

            with open(f"cache/{qa}.json", "w") as f:
                f.write(json.dumps(ndoc.to_json()))
            nvisited_items[qa] = ndoc

    print(nvisited_items.keys())


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


if __name__ == "__main__":
    main()
