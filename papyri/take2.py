import re
from textwrap import indent as _indent

import matplotlib
import numpy

lines = numpy.__doc__.split("\n")

lines = matplotlib.__doc__.split("\n")


ex = """
For the most part, direct use of the object-oriented library is encouraged when
programming; pyplot is primarily for working interactively. The exceptions are
the pyplot functions :dummy:`.pyplot.figure`, :domain:role:`.pyplot.subplot`, :also:dir:`.pyplot.subplots`,
and `.pyplot.savefig`, which `` can greatly simplify scripting. An example of verbatim code would be ``1+1 = 2`` but it is
not valid Python assign:: 
"""


WHAT = lambda x: "\033[96m" + x + "\033[0m"
HEADER = lambda x: "\033[95m" + x + "\033[0m"
BLUE = lambda x: "\033[94m" + x + "\033[0m"
GREEN = lambda x: "\033[92m" + x + "\033[0m"
ORANGE = lambda x: "\033[93m" + x + "\033[0m"
RED = lambda x: "\033[91m" + x + "\033[0m"
ENDC = lambda x: "\033[0m" + x + "\033[0m"
BOLD = lambda x: "\033[1m" + x + "\033[0m"
UNDERLINE = lambda x: "\033[4m" + x + "\033[0m"


class Node:
    def __init__(self, value):
        self.value = value

    @classmethod
    def parse(cls, tokens):
        return cls(tokens[0]), tokens[1:]

    def is_whitespace(self):
        if not isinstance(self.value, str):
            return False
        return not bool(self.value.strip())

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.value}>"


class Verbatim(Node):
    def __init__(self, value):
        self.value = value

    @classmethod
    def parse(cls, tokens):
        acc = []
        consume_start = None
        if len(tokens) < 5:
            return None
        if (tokens[0], tokens[1]) == ("`", "`") and tokens[2].strip():
            for i, t in enumerate(tokens[2:-2]):
                if t == "`" and tokens[i + 2] == "`":
                    return cls(acc), tokens[i + 4 :]
                else:
                    acc.append(t)
        return None

    @property
    def text(self):
        return "".join(self.value)

    def __len__(self):
        return sum(len(x) for x in self.value) + 4

    def __repr__(self):
        return RED("``" + "".join(self.value) + "``")


class Directive(Node):
    def __init__(self, value, domain, role):
        self.value = value
        self.domain = domain
        if domain:
            assert role
        self.role = role

    @property
    def text(self):
        return "".join(self.value)

    @classmethod
    def parse(cls, tokens):
        acc = []
        consume_start = None
        domain, role = None, None
        if tokens[0] == "`" and tokens[1] != "`" and tokens[1].strip():
            consume_start = 1
        elif (len(tokens) >= 4) and (tokens[0], tokens[2], tokens[3]) == (
            ":",
            ":",
            "`",
        ):
            domain, role = None, tokens[1]
            consume_start = 4
            pass
        elif len(tokens) >= 6 and (tokens[0], tokens[2], tokens[4], tokens[5]) == (
            ":",
            ":",
            ":",
            "`",
        ):
            domain, role = tokens[1], tokens[3]
            consume_start = 6

        if consume_start == None:
            return None

        for i, t in enumerate(tokens[consume_start:]):
            if t == "`":
                return cls(acc, domain, role), tokens[i + 1 + consume_start :]
            else:
                acc.append(t)

    def __len__(self):
        return sum(len(x) for x in self.value) + len(self.prefix)

    @property
    def prefix(self):
        prefix = ""
        if self.domain:
            prefix += ":" + self.domain
        if self.role:
            prefix += ":" + self.role + ":"
        return prefix

    def __repr__(self):
        prefix = ""
        if self.domain:
            prefix += ":" + self.domain
        if self.role:
            prefix += ":" + self.role + ":"
        # prefix = ''
        return GREEN(prefix) + HEADER("`" + "".join(self.value) + "`")


class Word(Node):
    def __repr__(self):
        return UNDERLINE(self.value)

    def __len__(self):
        return len(self.value)


def lex(lines):
    acc = ""
    for l in lines:
        for c in l:
            if c in " `*_:":
                if acc:
                    yield acc
                yield c
                acc = ""
            else:
                acc += c
        if acc:
            yield acc
            acc = ""
        yield " "


class FirstCombinator:
    def __init__(self, parsers):
        self.parsers = parsers

    def parse(self, tokens):
        for parser in self.parsers:
            res = parser.parse(tokens)
            if res is not None:
                return res

        return None


def rewrap(tokens, max_len):
    acc = [[]]
    clen = 0
    for t in tokens:
        if clen + (ll := len(t)) > max_len:
            # remove whitespace at EOL
            while acc and acc[-1][-1].is_whitespace():
                acc[-1].pop()
            acc.append([])
            clen = 0

        # do no append whitespace at SOL
        if clen == 0 and t.is_whitespace():
            continue
        acc[-1].append(t)
        clen += ll
    # remove whitespace at EOF
    try:
        while acc and acc[-1][-1].is_whitespace():
            acc[-1].pop()
    except IndexError:
        pass

    return acc


class Paragraph:
    def __init__(self, children, width=80):
        self.children = children
        self.width = width

    @classmethod
    def parse_lines(cls, lines):
        tokens = list(lex(lines))

        rest = tokens
        acc = []
        parser = FirstCombinator([Directive, Verbatim, Word])
        while rest:
            parsed, rest = parser.parse(rest)
            acc.append(parsed)

        return cls(acc)

    @property
    def references(self):
        refs = []
        for c in self.children:
            if isinstance(c, Directive) and c.role != "math":
                refs.append(c.text)
        return refs

    def __repr__(self):

        rw = rewrap(self.children, self.width)

        p = "\n".join(["".join(repr(x) for x in line) for line in rw])
        return f"""<Paragraph:\n{p}>"""


def indent(text, marker="   |"):
    """
    Return the given text indented with 3 space plus a pipe for display.
    """
    lines = text.split("\n")
    return "\n".join(marker + l for l in lines)


def is_at_header(lines):
    if len(lines) < 2:
        return False
    l0, l1, *rest = lines
    if len(l0.strip()) != len(l1.strip()):
        return False
    if len(s := set(l1.strip())) != 1:
        return False
    if next(iter(s)) in "-=":
        return True
    return False


def header_lines(lines):
    """
    Find lines indices for header
    """

    indices = []

    for i, l in enumerate(lines):
        if is_at_header(lines[i:]):
            indices.append(i)
    return indices


def separate(lines, indices):
    acc = []
    for i, j in zip([0] + indices, indices + [-1]):
        acc.append(lines[i:j])
    return acc


def with_indentation(lines, start_indent=0):
    """
    return pairs of indent_level and lines
    """

    indent = start_indent
    for l in lines:
        if (ls := l.lstrip()) :
            yield (indent := len(l) - len(ls)), l
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
    for i, l in with_indentation(lines):
        print(i, l)
        if not l.strip() and indent == start_indent:
            wh.append(l)
            continue
        else:
            if wh and i == indent == start_indent:
                acc.append(Block(blk, wh, [], reason="wh+re0"))
                blk = [l]
                wh = []
            else:
                blk.extend(wh)
                wh = []
                blk.append(l)

            indent = i
    acc.append(Block(blk, wh, [], reason="end"))
    return acc


def eat_while(lines, condition):
    acc = []
    for i, l in enumerate(lines):
        if condition(l):
            acc.append(l)
            continue
        break
    else:
        return acc, []
    return acc, lines[i:]


def make_blocks_2(lines):
    """
    WRONG:

    xxxxx

    yyyyyy

       zzzzz

    ttttttttt

    x and y should be 2blocks

    """
    if not lines:
        return []
    l0 = lines[0]

    ind0 = len(l0) - len(l0.lstrip())

    rest = lines
    acc = []
    while rest:
        blk, rest = eat_while(rest, lambda l: len(l) - len(l.lstrip()) == ind0)
        wht, rest = eat_while(rest, lambda l: not l.strip())
        ind, rest = eat_while(
            rest, lambda l: ((len(l) - len(l.lstrip())) > ind0 or not l.strip())
        )
        acc.append(Block(blk, wht, ind))

    return acc


def make_block_3(lines: "Lines"):
    """
    I think the correct alternative is that each block may get an indented children, and that a block is thus: 

    - a) The sequence of consecutive non blank lines with 0 indentation
    - b) The (potentially absent) blank lines leading to the indent block
    - c) The Raw indent block (we can decide to recurse, or not later)
    - d?) The trailing blank line at the end of the block leading to the next one. I think the blank line will be in the
      raw indent block

    """

    (a, b, c) = [], [], []
    blocks = []
    state = "a"

    for l in lines:
        if l.indent == 0:
            if state == "a":
                a.append(l)
            elif state in ("b", "c"):
                blocks.append((a, b, c))
                a, b, c = [l], [], []
                state = "a"
            else:
                raise ValueError
        elif l.indent == None:
            if state == "a":
                state = "b"
                b.append(l)
            elif state == "b":
                b.append(l)
            elif state == "c":
                c.append(l)
            else:
                raise ValueError
        elif l.indent > 0:
            if state in ("a", "b"):
                state = "c"
                c.append(l)
            elif state == "c":
                c.append(l)
            else:
                raise ValueError

    blocks.append((a, b, c))
    return blocks


class Block:
    """
    The following is wrong for some case, in particular if there are many paragraph in a row with 0 indent. 
    we can't ignore blank lines.

    --- 

    A chunk of lines that breaks when the indentation reaches 
    - the last of a list of blank lines if indentation is consistant
    - the last non-0  indented lines


    Note we likely want the _body_ lines and then the _indented_ lines if any, which would mean we
    cut after the first blank lines and expect indents, otherwise there is not indent.
    and likely if there is a blank lnes as  a property.

    ---- 

    I think the correct alternative is that each block may get an indented children, and that a block is thus: 

    - 1) The sequence of consecutive non blank lines with 0 indentation
    - 2) The (potentially absent) blank lines leading to the indent block
    - 3) The Raw indent block (we can decide to recurse, or not later)
    - 4) The trailing blank line at the end of the block leading to the next one.

    """

    def __init__(self, lines, wh, ind, *, reason=None):
        self.lines = lines
        self.wh = wh
        self.ind = make_blocks_2(ind)
        self.reason = reason

    def __repr__(self):
        return (
            f"<Block body-len='{len(self.lines)},{len(self.wh)},{self.reason}'> with\n"
            + indent("\n".join(self.lines + self.wh), "    ")
            + "\n"
            + indent("\n".join([repr(x) for x in self.ind]), "    ")
        )


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
            return make_blocks_2(self.lines[2:])
        else:
            return make_blocks_2(self.lines)

    def __repr__(self):
        return (
            f"<Section header='{self.header[0]}' body-len='{len(self.lines)}'> with\n"
            + indent("\n".join([str(b) for b in self.body]) + "...END\n\n", "    |")
        )


# wrapper around handling lines


class Line:
    def __init__(self, line, number, offset=0):
        self._line = line
        self._number = number
        self._offset = offset

    @property
    def text(self):
        return self._line.rstrip()

    @property
    def blank(self):
        return self._line.strip() == ""

    def __getattr__(self, missing):
        return getattr(self._line, missing)

    def __repr__(self):
        return f"<Line {self._number: 3d} {str(self.indent):>4}| {self._line[self._offset:]}>"

    @property
    def indent(self):
        if self.blank:
            return None
        return len(self._line) - len(self._line.lstrip()) - self._offset


class Lines:
    def __init__(self, lines):
        self._lines = [
            l if isinstance(l, Line) else Line(l, n) for n, l in enumerate(lines)
        ]

    def __getitem__(self, sl):
        if isinstance(sl, int):
            return self._lines[sl]
        else:
            return Lines(self._lines[sl])

    def __repr__(self):
        rep = f"<Lines {len(self._lines)} lines:"
        for l in self._lines:
            rep += f"\n    {l}"
        rep += ">"
        return rep

    def dedented(self):
        pass


class Document:
    def __init__(self, lines):
        self.lines = lines

    @property
    def sections(self):
        indices = header_lines(self.lines)
        return [Section(l) for l in separate(self.lines, indices)]

    def __repr__(self):
        acc = ""
        for i, s in enumerate(self.sections[0:]):
            acc += "\n" + repr(s)
        return "<Document > with" + indent(acc)


# d = Document(lines[:])
# for i, l in with_indentation(repr(d).split("\n")):
#    print(i, l)

if __name__ == "__main__":
    ex = matplotlib.__doc__
    import numpy as np

    ex = np.__doc__
    for b in make_block_3(Lines(ex.split("\n"))[:]):
        for m, u in zip("| >", b):
            for x in u:
                print(m, x._line)
    # for b in with_indentation(ex.split('\n')):
    #    print(b)
    # print(ex)
    # for w in [80, 120]:
    #    p = Paragraph.parse_lines(ex.split("\n"))
    #    p.width = w
    #    print(p)
    #    print()
