import numpy as np
import scipy
import scipy.special
import sklearn
from velin import parsedoc, NumpyDocString
#from numpydoc.docscrape import NumpyDocString
from types import ModuleType

import matplotlib
import matplotlib.pyplot
import inspect

import numpy.core.numeric

modules = [
    np,
#    np.fft,
#    scipy,
    scipy.special,
#    numpy.core.numeric,
#    sklearn,
#    matplotlib,
#    matplotlib.pyplot,
]

from velin.ref import NumpyDocString

from velin.examples_section_utils import splitblank, splitcode, InOut
from textwrap import dedent
def dedent_but_first(text):
    a,*b = text.split('\n')
    return dedent(a)+'\n'+dedent('\n'.join(b))

from jinja2 import Environment, PackageLoader, select_autoescape
from jinja2 import ChoiceLoader, FileSystemLoader
env = Environment(
    loader=FileSystemLoader('velin'),
    autoescape=select_autoescape(['html', 'xml', 'tpl', 'tpl.j2'])
)

template = env.get_template('core.tpl.j2')

from pygments.lexers import PythonLexer
import jedi
def pos_to_nl(script, pos):
    rest = pos
    ln=0
    for line in script.splitlines():
        if len(line) < rest:
            rest -= len(line)+1
            ln +=1
        else:
            return ln, rest
            
P = PythonLexer()
def parse_script(script, ns=None):
    
    
    jeds = []
    if ns:
        jeds.append(jedi.Interpreter(script, namespaces=[ns]))
    jeds.append(jedi.Script(script))




    for index, type_, text in P.get_tokens_unprocessed(script):
        a,b = pos_to_nl(script, index)
        try:
            ref = None
            for jed in jeds:
                try:
                    ref = jed.infer(a+1,b)[0].full_name
                except (AttributeError, TypeError, Exception):
                    pass
                break
        except IndexError:
            ref = ''
        yield index, type_, text, ref     

def get_example_data(doc):
    blocks = list(map(splitcode, splitblank(doc['Examples'])))
    edata = []
    for b in blocks:
        for item in b:
            if isinstance(item, InOut):
                script = '\n'.join(item.in_)
                entries = list(parse_script(script, ns={'np':np}))
                edata.append(['code', ( entries, '\n'.join(item.out))])

            else:
                edata.append(['text','\n'.join(item.out)])
    return edata


visited_items = {}
nvisited_items = {}

for mod in modules:
    if not mod.__name__.startswith(("num", "sci", "skl", "mat", 'core')):
        #print('skip', mod)
        #continue
        pass
    # print('exploring module', mod)
    for n in dir(mod) :
        if n == "ufunc":
            continue
        try:
            a = getattr(mod, n)
        except Exception:
            continue
        if isinstance(a, ModuleType):
            if a not in modules:
                pass
                #modules.append(a)
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
        if  (ddd := getattr(a, "__doc__", None)) is None:
            # print('no doc for', a)
            continue
        #sig = None
        #try:
        #    sig = str(inspect.signature(a))
        #except (ValueError, TypeError):
        #    pass
        try:
            #doc, warnings = parsedoc(a.__doc__, name=qa, sig=sig)
            ndoc = NumpyDocString(dedent_but_first(ddd))
        except:
            print('failed', a)
            raise
            continue
        #if warnings:
        #    print(qa)
        #    for w in warnings:
        #        print("  |", w)
        nsa = ndoc['See Also']
        refs = []
        if nsa:
            for line in nsa:
                rt, desc = line
                for ref, type_ in rt:
                #ref, type_ = rt
                    refs.append(ref)
                    # ?assert type_ is None
                    #print( '   >', ref)
                #print( '   >    ', desc)
        #sa = doc.see_also()
        #left, right = set(sa)-set(refs), set(refs)-set(sa)
        #if left or right:
        #    print('   :', left, right)
        #    print('   :', nsa)
        #    print('   :', a.__doc__)
        if getattr(visited_items, qa, None):
            raise ValueError(f"{qa} already visited")
        #visited_items[qa] = doc
        ndoc.backrefs = []
        ndoc.edata = get_example_data(ndoc)
        #if 'Examples' in a.__doc__:
        #    print(a, qa, 'has examples')
        #if ndoc['Examples']:
        #    print(ndoc.edata, qa)
        ndoc.refs = list({u[3] for t_,sect in ndoc.edata if t_ == 'code' for u in sect[0] if u[3]})
        ndoc.backrefs = []
        print(qa)
        nvisited_items[qa] = ndoc
        # if sa:
        # print(qa)
        # print(sa)
        # with open(f'html/{qa}.html','w') as f:
        #    f.write(s)

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



for qa, ndoc in nvisited_items.items():
    for ref in ndoc.refs:
        print('trying', qa, '<-', ref)
        if (ref) in nvisited_items and ref != qa:
            nvisited_items[ref].backrefs.append(qa)

for qa, ndoc in nvisited_items.items():
    #ndoc = nvisited_items[qa]
    #sa = doc.see_also()
    nsa = ndoc.refs
    #if not sa:
    #    continue
    #for backref in sa:
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
    #s = doc._repr_html_(lambda ref: resolver(qa, visited_items, ref))
    #ndoc = nvisited_items[qa]
    
    with open(f"html/{qa}.html", "w") as f:
        f.write(template.render(doc=ndoc))
