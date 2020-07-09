
import numpy as np
import scipy
from minirst import parsedoc
from types import ModuleType

modules=[np, np.fft, scipy ]
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

        #print(qa)
        s = (doc := parsedoc(a.__doc__))._repr_html_()
        doc
        doc.see_also()
        with open(f'html/{qa}.html','w') as f:
            f.write(s)
