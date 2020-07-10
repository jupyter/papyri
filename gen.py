
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

        if hasattr(a, '__qualname__'):
            qa = mod.__name__+'.'+a.__qualname__
        else:
            qa = mod.__name__+'.'+n
            #print('skipping', type(a), getattr(a, '__qualname__', None), f'({n}?)')
            #continue
        if getattr(a, '__doc__', None) is None:
            #print('no doc for', a)
            continue
        if isinstance(a, ModuleType):
            print('skip module', a)
            continue

        s = (doc := parsedoc(a.__doc__))._repr_html_()
        doc
        sa = doc.see_also()
        if getattr(visited_items, qa, None):
            raise ValueError(f'{qa} already visited')
        visited_items[qa] = doc
        #if sa:
            #print(qa)
            #print(sa)
        #with open(f'html/{qa}.html','w') as f:
        #    f.write(s)

print(visited_items.keys())

for qa,doc in visited_items.items():
    sa = doc.see_also()
    for backref in sa:
        if backref in visited_items:
            #print('easy: 1', qa, '->', backref)
            continue
        elif 'numpy.'+backref in visited_items:
            #print('easy: 2', qa, '->', backref)
            continue

        root,_ = qa.rsplit('.',1)

        if root+'.'+backref in visited_items:
            print('easy: 3', qa, '->', backref)
            pass
        else:
            print('???', qa, '-?>', backref)
