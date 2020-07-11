import numpy as np
import difflib

file = np.__file__
file = '/Users/bussonniermatthias/dev/numpy/numpy/matlib.py'
with open(file, 'r') as f:
    data = f.read()
    
import ast
tree = ast.parse(data)

funcs = [t for t in tree.body if isinstance(t, ast.FunctionDef)]

import difflib

def w(orig):
    ll = []
    for l in orig:
        if l[0] in '+-':
            #ll.append(l.replace(' ', '⎵'))
            ll.append(l.replace(' ', '·'))
            
        else:
            ll.append(l)
    lll = []
    for l in ll:
        if l.endswith('\n'):
            lll.append(l[:-1])
        else:
            lll.append(l[:])
    return lll
import numpydoc.docscrape
from textwrap import indent

class DocstringFormatter:

    @classmethod
    def format_Summary(self, s):
        return '\n'.join(s)+'\n'

    @classmethod
    def format_Extended_Summary(self, es):
        return '\n'.join(es)+'\n'

    @classmethod
    def _format_ps(cls, name, ps):
    
        out  = name+'\n'
        out += '-'*len(name)+'\n'
        for p in ps:
            if p.type:
                out+= f'''{p.name} : {p.type}\n'''
            else:
                out+= f'''{p.name}\n'''
            out+= indent('\n'.join(p.desc), '    ')
            out+= '\n'
        return out   
    
    @classmethod
    def format_Parameters(cls, ps):
        return cls._format_ps('Parameters', ps)
        
    
    @classmethod
    def format_Other_Parameters(cls, ps):
        return cls._format_ps('Other Parameters', ps)
    
    @classmethod
    def format_See_Also(cls, sas):
        out  = 'See Also\n'
        out  +='--------\n'
        
        for a,b in sas:
            if b:
                out+='Some See also B !\n'
            if len(a) > 1:
                out+='Some See Long A !\n'
            ref, type_ = a[0]
            if type_ is not None:
                out += f':{type_}:`{ref}`\n'
            else:
                out += f'{ref}\n'
        return out
    
    @classmethod
    def format_Notes(cls, lines):
        out  = 'Notes\n'
        out += '-----\n'
        out+= '\n'.join(lines)
        out+= '\n'
        return out   
    
    @classmethod
    def format_Examples(cls, lines):
        out  = 'Examples\n'
        out += '--------\n'
        out+= '\n'.join(lines)
        out+= '\n'
        return out  
        
    @classmethod
    def format_Returns(self, ps):
        out  = 'Returns\n'
        out  += '-------\n'

        for p in ps:
            if p.name:
                out+= f'''{p.name} : {p.type}\n'''
            else:
                out+= f'''{p.type}\n'''
            if p.desc:
                out+= indent('\n'.join(p.desc), '    ')
                out+= '\n'
        return out    
        

        
def test(docstr):
    if len(docstr.splitlines()) == 1:
        return
    if not docstr.startswith('\n    '):
        docstr = '\n    '+docstr
    doc = numpydoc.docscrape.NumpyDocString(docstr)


    fmt = ''
    for s in doc.sections:
        if doc[s]:
            f = getattr(DocstringFormatter, 'format_'+s.replace(' ', '_'), lambda x:s)
            fmt += '\n'
            fmt += f(doc[s])
    
    fmt = indent(fmt, '    ')+'    '
    #print(fmt)

    dold = docstr.splitlines()
    dnew = fmt.splitlines()
    diffs = list(difflib.unified_diff(dold, dnew,n=100, fromfile='old',tofile='new'), )
    if diffs:
        
        print(indent('\n'.join(w(diffs)), ' |   ', predicate=lambda x:True))

for i,func in enumerate(funcs[:]):
    print(i,'==', func.name, '==')
    try:
        docstring = func.body[0].value.s
    except AttributeError:
        continue
    (func.body[0].lineno,
    func.body[0].col_offset,
    func.body[0].end_lineno)

    test(docstring)
