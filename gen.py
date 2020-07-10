
import numpy as np
import scipy
from minirst import parsedoc
from types import ModuleType

modules=[np, np.fft, np.ndarray]

visited_items = {}

for mod in modules:
    for n in dir(mod):
        if n == 'ufunc':
            continue
        a = getattr(mod, n)
        if getattr(a, '__module__', None) is None:
            continue
        if hasattr(a, '__qualname__'):
            qa = a.__module__+'.'+a.__qualname__
        else:
            qa = a.__module__+'.'+n
            #print('skipping', type(a), getattr(a, '__qualname__', None), f'({n}?)')
            #continue
        if qa.startswith('nd'):
            print(qa, mod, mod.__name__)
        if getattr(a, '__doc__', None) is None:
            #print('no doc for', a)
            continue
        if isinstance(a, ModuleType):
            #print('skip module', a)
            continue

        doc = parsedoc(a.__doc__)
        sa = doc.see_also()
        if getattr(visited_items, qa, None):
            raise ValueError(f'{qa} already visited')
        visited_items[qa] = doc
        #if sa:
            #print(qa)
            #print(sa)
        #with open(f'html/{qa}.html','w') as f:
        #    f.write(s)

#print(visited_items.keys())

def _a(qa, backref):
    return backref

def _b(qa, backref):
    root,_ = qa.rsplit('.',1)
    return root+'.'+backref


for qa,doc in visited_items.items():
    sa = doc.see_also()
    if not sa:
        continue
    for backref in sa:
        for x in _a, _b:
            br = x(qa, backref)
            if br in visited_items:
                visited_items[br].backrefs.append(qa)
                #print(br, '<-', qa)
                break
        else:
            #print('???', qa, '-?>', backref)
            pass

for qa,doc in visited_items.items():
    s = doc._repr_html_()
    with open(f'html/{qa}.html','w') as f:
            f.write(s)

