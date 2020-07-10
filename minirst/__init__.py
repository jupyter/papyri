"""
"""

import sys
from pathlib import Path
import textwrap

__version__ = '0.0.1'

def parse(input):
    """
    parse an input string into token/tree.

    For now only return a list of tokens

    """
    tokens=[]
    for l in input.splitlines():
        tokens.extend(l.split(' '))
    return tokens


def transform(tokens):
    """
    Accumulate tokens in lines.

    Add token (and white spaces) to a line until it overflow 80 chars.
    """
    lines = []
    current_line = []
    for t in tokens:
        if sum([len(x)+1 for x in current_line])+len(t) > 80:
            lines.append(current_line)
            current_line=[]
        current_line.append(t)
    if current_line:
        lines.append(current_line)
    return lines


def format(lines):
    s = ''
    for l in lines:
        s = s+ ' '.join(l)
        s = s+'\n'
    return s[:-1]


def compute_indents(lines):
    """
    Given a list of lines, compute indentation in number of spaces.

    Indentation is only supported if spaces, tabs raise a NotImplementedError as we don't know how wide a tab is.
    We also treat complete blank lines as `None` indentation.


    """
    assert isinstance(lines, list)
    results = []
    for l in lines:
        s = l.lstrip()
        if not s:
            results.append(None)
            continue
        indent = len(l) - len(s)
        if '\t' in l[:indent]:
            raise NotImplementedError

        results.append(indent)
    return results


class TryNext(Exception):pass

class Header:

    def __init__(self, title, level):
        self.title = title
        self.level = level

    def __repr__(self):
        return f'<Header {self.level}:  {self.title} >'

    def __str__(self):
        tt = str(self.title)
        return  tt + '\n' + '='*len(tt)

    def _repr_html_(self):
        return '<h1>'+str(self.title)+'</h1>'

    @classmethod
    def parse(cls, lines):
        if len(lines) < 2:
            raise TryNext
        assert lines
        title, _ = Raw.parse(lines)
        lgth = len(title.lines[0])
        if len(set(lines[1])) == 1 and len(lines[1]) !=1 and len(lines[1]) != lgth:
            print('======= WRONG LEN? ======')
            print(lines[0])
            print(lines[1])
        level = allunders(lines[1], lgth)
        return cls(title, level), lines[2:]

def allunders(line, lenght):

    if not len(set(line)) == 1 or not len(line) == lenght:
        raise TryNext
    return 0




class Any:

    def __init__(self, line):
        assert isinstance(line, str)
        self.lines = [line]

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.lines!r} >'

    def __str__(self):
        return self.lines[0]

    @classmethod
    def parse(cls, lines):
        return cls(lines[0]), lines[1:]

class Raw(Any):pass


class RawTilNextHeader:

    @classmethod
    def parse(cls, lines):
        ll = []
        for i,l in enumerate(lines):
            try:
                Header.parse(lines[i:])
                break
            except TryNext:
                pass

            ll.append(l)
        return cls(ll), lines[i+1:]

    def __init__(self, items):
        assert isinstance(items, list)
        self.items = items


    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.items!r} >'

    def __str__(self):
        return '\n'.join(self.items)

    def _repr_html_(self):
        return """<pre>"""+ str(self)+ """</pre>"""


class DescriptionList:

    @classmethod
    def parse(cls, lines):
        dct ={}
        key, values = None, []
        for i,l in enumerate(lines):
            try:
                Header.parse(lines[i:])
                dct[key] = values
                break
            except TryNext:
                pass

            if not l.startswith(' '):
                dct[key] = values
                key, values = l.strip(), []
                if ',' in key:
                    raise TryNext
            else:
                values.append(l)
        del dct[None]
        return cls(dct), lines[i:]

    def __init__(self, items):
        assert isinstance(items, dict)
        self.items = items

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.items!r} >'

    def __str__(self):
        def _f(v):
            return '\n'.join(v)

        return '\n'.join(
            f"{k}:\n{_f(v)}" for k,v in self.items.items()
        )

    def _repr_html_(self):
        def _f(v):
            return '\n'.join(v)
        return """<dl>"""+ '\n'.join(f"<dt>{k}</dt>\n<dd>{_f(v)}</dd>" for k,v in self.items.items())+ """</dl>"""
        
class Listing:

    @classmethod
    def parse(cls, lines):
        assert ',' in lines[0] , lines[0]
        listing = [l.strip() for l in lines[0].split(',')]
        return cls(listing), lines[1:]


    def __init__(self, listing):
        self.listing = listing

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.listing!r} >'


    def __str__(self):
        return ','.join([f"{k}" for k in self.listing])

    def _repr_html_(self):
        return """<ul>"""+ '\n'.join(f"<li>{k}</li>" for k in self.listing)+ """</ul>"""


class Mapping:

    @classmethod
    def parse(cls, lines):
        mapping = {}
        k = None
        for i,l in enumerate(lines):
            if not l.strip():
                continue
            try:
                Header.parse(lines[i:])
                break
            except TryNext:
                pass
            if l.startswith(' ') and k:
                mapping[k.strip()] += l.strip()
                continue
            if ':' in l:
                k,v = l.split(':', maxsplit=1)
                mapping[k.strip()] = v
            elif ',' not in l:
                k,v = l.strip(), None
                mapping[k] = v
            else:
                for k in l.split(','):
                    mapping[k.strip()] = None


        return cls(mapping), lines[i:]


    def __init__(self, mapping):
        self.mapping = mapping

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.mapping!r} >'


    def __str__(self):
        return '\n'.join([f"{k}: {v}" for k,v in self.mapping.items()])

    def _repr_html_(self):
        return """<dl>"""+ '\n'.join(f"<dt>{k}</dt>\n<dd>{v}</dd>" for k,v in self.mapping.items())+ """</dl>"""




class CodeBlock:

    @classmethod
    def parse(cls, lines):
        if not lines[0].startswith(('>>>', '    >>>')):
            raise TryNext
        
        _lines = []
        for i,l in enumerate(lines):
            if not l.strip():
                break
            _lines.append(l)
        else:
            return cls(_lines), []
        return cls(_lines), lines[i:]


        
    def __init__(self, lines):
        self.lines = lines


    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.lines!r} >'

    def _repr_html_(self):
        return '<pre>'+'\n'.join(self.lines)+'</pre>'


    def __str__(self):
        return '\n'.join(self.lines) 





            

class Doc:


    @classmethod
    def parse(cls, lines):
        parsed = []
        cl = len(lines)
        while lines:
            for t in Section.parse, Header.parse, BlankLine.parse, CodeBlock.parse, Paragraph.parse, RawTilNextHeader.parse, failed:
                try:
                    node, lines_ = t(lines)
                    if isinstance(node, list):
                        parsed.extend(node)
                    else:
                        parsed.append(node)
                    if len(lines_) >= len(lines):
                        raise ValueError('Could not parse', lines)
                    lines = lines_
                    break
                except TryNext:
                    pass
        return cls(parsed)

    def __init__(self, nodes):
        self.nodes = nodes

    def __repr__(self):
        return f'<{self.__class__.__name__}\n'+'\n'.join([textwrap.indent(repr(n), '    ') for n in self.nodes])+ '\n>'

    def see_also(self):
        for i,p in enumerate(self.nodes):
            if isinstance(p, Section) and p.header.title.lines[0] == 'See Also':
                break
        else:
            return []

        node = self.nodes[i+1]
        if isinstance(node, Mapping):
            return node.mapping.keys()
        else:
            print('not a mapping', node)



    def _repr_html_(self):
        base = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="https://fonts.xz.style/serve/inter.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@exampledev/new.css@1.1.2/new.min.css">
</head>
<body>
    {}
</body>
</html>
"""

        return base.format('\n'.join(n._repr_html_() for n in self.nodes))





class Section:
    """
    start with a header, but have custom parsing because we know about it in numpydoc.
    """

    @classmethod
    def parse(cls, lines):
        header, rest = Header.parse(lines)
        if header.title.lines[0] in ('Parameters', 'Returns', 'Attributes', 'Raises', 'Methods', 'Warns', 'Other Parameters'):
            core, rest = DescriptionList.parse(rest)
            return [cls(header), core], rest
        elif header.title.lines[0] in ('See Also', 'Returns', 'See also'):
            if header.title.lines[0] == 'See also':
                header.title.lines[0] = 'See Also'
            try:
                core, rest = Mapping.parse(rest)
            except TryNext:
                core, rest = DescriptionList.parse(rest)

            return [cls(header), core], rest
        elif header.title.lines[0] in ('Examples',):
            return [cls(header)], rest
        elif header.title.lines[0] in ('Notes', 'References'):
            return [cls(header)], rest

        else:
            raise ValueError(header.title.lines[0])

    def __init__(self, header):
        self.header = header


    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.header!r}>'

    def __str__(self):
        return str(self.header)

    def _repr_html_(self):
        return self.header._repr_html_()
            




class Paragraph:

    @classmethod
    def parse(cls, lines):
        _lines = []
        l0 = lines[0]
        if not l0 or l0.startswith(' '):
            raise TryNext
        for i,l in enumerate(lines):
            if l and not l.startswith(' '):
                _lines.append(l)
            else:
                break
        if not _lines:
            raise TryNext
        return cls(_lines), lines[i+1:]

    def __init__(self,lines):
        self.lines = lines

    def __repr__(self):
        return  f"<{self.__class__.__name__}: {self.lines}>"

    def __str__(self):
        return '\n'.join(self.lines)

    def _repr_html_(self):
        return '<p>'+' '.join(self.lines)+'</p>'


class BlankLine:

    @classmethod
    def parse(self, lines):
        if not lines[0].strip():
            return BlankLine(), lines[1:]
        raise TryNext

    def __init__(self):
        pass

    def __repr__(self):
        return  f"<{self.__class__.__name__}>"

    def __str__(self):
        return ''

    def _repr_html_(self):
        return ''

def failed(lines):
    raise ValueError('nothign managed to parse', lines)

def parsedoc(doc):

    lines = dedentfirst(doc).splitlines()
    return Doc.parse(lines)









def find_indent_blocks(lines):
    """
    Given a list of lines find _block_ by level of indentation

    - A block is considered a sequence of one or more lines, separated 
    - once one level of indentation is encounter we don't split anymore, it will be up to the sub-parser.
    - A whitespace or empty line yield another block.
    """
    if isinstance(lines, str):
        raise ValueError('split please')
    indents = compute_indents(lines)
    assert len(indents) == len(lines)
    l0 = lines[0]
    indent_level = indents[0]
    if indent_level is None:
        indent_level = 0
    assert indent_level == 0
    current_block = [l0]

    n_empty_lines = 0

    blocks = []
    for new_level,l in zip(indents[1:],lines[1:]):
        if new_level is None:
            n_empty_lines +=1
            continue

        if n_empty_lines and new_level == 0:
            current_block.extend(['']*n_empty_lines)
            n_empty_lines = 0
            blocks.append((indent_level, current_block))
            current_block = [l]
            indent_level = new_level
            continue
        if n_empty_lines:
           current_block.extend(['']*n_empty_lines)
           n_empty_lines = 0

        if indent_level == 0 and new_level:
            # start a new block
            blocks.append((0, current_block))
            current_block = [l[new_level:]]
            indent_level = new_level
            continue


        # we are already in indented blocks.
        if new_level >= indent_level:
            current_block.append(l[indent_level:])
        elif new_level < indent_level:
            blocks.append((indent_level, current_block))
            current_block = [l]
            indent_level = new_level
            continue
    if current_block:
        blocks.append((indent_level, current_block))
        current_block = []

    assert len(current_block) == 0
    return blocks





def dedentfirst(docstring):
    from textwrap import dedent
    lines = docstring.splitlines()
    ln = dedent('\n'.join(lines[1:])).splitlines()
    l0 = lines[0]
    return '\n'.join([dedent(l0)]+ln)





def reformat(input):
    return format(transform(parse(format(transform(parse(input))))))

def main():
    filename = sys.argv[1]
    p = Path(filename)

    with p.open() as f:
        data = f.read()
        print(reformat)
    


if __name__ == '__main__':
    main()
