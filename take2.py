import numpy 
import matplotlib

lines = numpy.__doc__.split('\n')

lines = matplotlib.__doc__.split('\n')
from textwrap import indent as _indent

def indent(text, ind='    '):
    return _indent(text, ind, lambda x:True)

def is_at_header(lines):
    if len(lines) < 2:
        return False
    l0,l1,*rest = lines
    if len(l0.strip()) != len(l1.strip()):
        return False
    if len(s := set(l1.strip())) != 1:
        return False
    if next(iter(s)) in '-=':
        return True
    return False


def header_lines(lines):
    """
    Find lines indices for header
    """
    
    indices = []

    for i,l in enumerate(lines):
        if is_at_header(lines[i:]):
            indices.append(i)
    return indices

def separate(lines, indices):
    acc = []
    for i,j in zip([0]+indices, indices+[-1]):
        acc.append(lines[i:j])
    return acc



def with_indentation(lines, start_indent=0):
    """
    return pairs of indent_level and lines
    """

    indent = start_indent
    for l in lines:
        if (ls := l.lstrip()):
            yield (indent := len(l) - len(ls)),l
        else:
            yield indent, l


def make_blocks(lines, start_indent=0):
    l0 = lines[0]
    start_indent = len(l0) - len(l0.lstrip())
    indent = start_indent
    acc = []
    blk = []
    wh = []
    reason = None
    for i,l in with_indentation(lines):
        if not l.strip() and indent==start_indent:
            wh.append(l)
            continue
        else:
            #if i<indent:
            #    acc.append(Block(blk, wh, reason=))
            #    blk = [l]
            #    wh = []
            if wh and i==indent==start_indent:
                acc.append(Block(blk, wh, reason='wh+re0'))
                blk = [l]
                wh = []
            else:
                blk.append(l)
                
            indent = i
    acc.append(Block(blk, wh, 'end'))
    return acc

def make_blocks_2(lines):
    acc = []

    blk = []
    ind = []
    for i,l in with_indentation(lines):pass
        
    return acc



class Block:
    """
    A chunk of lines that breaks when the indentation reaches 
    - the last of a list of whitelines if indentation is consistant
    - the last non-0  indented lines


    Note we likely want the _body_ lines and then the _indented_ lines if any, which would mean we
    cut after the first whitespace lines and expect indents, otherwise there is not indent.
    and likely if there is a whiteline as  a property.
    """

    def __init__(self, lines, wh, reason):
        self.lines = lines
        self.wh = wh
        self.reason = reason
    def __repr__(self):
        return f"<Block body-len='{len(self.lines)},{len(self.wh)},{self.reason}'> with\n"\
                +indent('\n'.join(self.lines+self.wh), '   |')


class Section:
    """
    A section start (or not) with a header.

    And have a body
    """

    def __init__(self, lines):
        self.lines = lines

    @property
    def header(self):
        if is_at_header(self.lines):
            return self.lines[0:2]
        else:
            return None, None

    @property
    def body(self):
        if is_at_header(self.lines):
            return make_blocks(self.lines[2:])
        else:
            return make_blocks(self.lines)

    def __repr__(self):
        return f"<Section header='{self.header[0]}' body-len='{len(self.body)}'> with\n"\
                +indent('\n'.join([str(b) for b in self.body])+'...END\n\n', '    |')





class Document:

    def __init__(self, lines):
        self.lines =  lines
        

    @property
    def sections(self):
        indices = header_lines(self.lines)
        return [Section(l) for l in separate(self.lines, indices)]

    def __repr__(self):
        acc = '' 
        for i,s in enumerate(self.sections[0:5]):
            acc+='\n'+repr(s)
        return '<Document > with'+indent(acc)






d = Document(lines[:40])
for i,l in with_indentation(repr(d).split('\n')):
    print(i,l)
