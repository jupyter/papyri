"""
Attempt at a multi-pass CST (concrete syntax tree) RST-ish parser.

This does not (and likely will not) support all of RST syntax, and may support
syntax that is not in the rst spec, mostly to support Python docstrings.

Goals
-----

The goal in here is to parse RST while keeping most of the original information
available in order to be able to _fix_ some of them with minimal of no changes
to the rest of the original input. This include but not limited to having
consistent header markers, and whether examples are (or not) indented with
respect to preceding paragraph.

The second goal is flexibility of parsing rules on a per-section basis;
Typically numpy doc strings have a different syntax depending on the section you
are in (Examples, vs Returns, vs Parameters), in what looks like; but is not;
definition list.

This also should be able to parse and give you a ast/cst without knowing ahead
of time the type of directive that are registered.

This will likely be used in the project in two forms, a lenient form that try to
guess as much as possible and suggest update to your style.

A strict form that avoid guessing and give you more, structured data.


Implementation
--------------

The implementation is not meant to be efficient but works in many multiple pass
that refine the structure of the document in order to potentially swapped out
for customisation.

Most of the high level split in sections and block is line-based via the
Line/lines objects that wrap a ``str``, but keep track of the original line
number and indent/dedent operations.


Junk Code
---------

There is possibly a lot of junk code in there due to multiple experiments.

Correctness
-----------

Yep, many things are probably wrong; or parsed incorrectly;

When possible if there is an alternative way in the source rst to change the
format, it's likely the way to go.

Unless your use case is widely adopted it is likely not worse the complexity
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from there import print

from papyri.utils import dedent_but_first

# 3x -> 9x for bright
WHAT = lambda x: "\033[36m" + x + "\033[0m"
HEADER = lambda x: "\033[35m" + x + "\033[0m"
BLUE = lambda x: "\033[34m" + x + "\033[0m"
GREEN = lambda x: "\033[32m" + x + "\033[0m"
ORANGE = lambda x: "\033[33m" + x + "\033[0m"
RED = lambda x: "\033[31m" + x + "\033[0m"
ENDC = lambda x: "\033[0m" + x + "\033[0m"
BOLD = lambda x: "\033[1m" + x + "\033[0m"
UNDERLINE = lambda x: "\033[4m" + x + "\033[0m"


from papyri.miniserde import deserialize, get_type_hints, serialize


class Base:
    @classmethod
    def _instance(cls):
        return cls()

    @classmethod
    def _deserialise(cls, **kwargs):
        # print("will deserialise", cls)
        try:
            instance = cls._instance()
        except Exception as e:
            raise type(e)(f"Error deserialising {cls}, {kwargs})") from e
        for k, v in kwargs.items():
            setattr(instance, k, v)
        return instance


class Node(Base):
    def __init__(self, value=None):
        self.value = value

    def __eq__(self, other):
        if not (type(self) == type(other)):
            return False
        tt = get_type_hints(type(self))
        for attr in tt:
            a, b = getattr(self, attr), getattr(other, attr)
            if a != b:
                return False

        return True

    @classmethod
    def _instance(cls):
        return cls()

    @classmethod
    def parse(cls, tokens):
        """
        Try to parse current `tokens` stream from current position.

        Returns
        -------
        Tuple with the following items:
            - Node to insert at current position in the token tree;
            - None if could not parse.

        """
        return cls(tokens[0]), tokens[1:]

    def is_whitespace(self):
        if not isinstance(self.value, str):
            return False
        return not bool(self.value.strip())

    def __repr__(self):
        return f"<{self.__class__.__name__}: \n{indent(repr(self.value))}>"

    def to_json(self):
        return serialize(self, type(self))

    @classmethod
    def from_json(cls, data):
        return deserialize(cls, cls, data)


@dataclass(frozen=True)
class RefInfo(Node):
    """
    This is likely not completely correct for target that are not Python object,
    like example of gallery.

    We also likely want to keep a reference to original object for later updates.


    Parameters
    ----------
    module:
        the module this object is defined in
    version:
        the version of the module where this is defined in
    kind: {'api', 'example', ...}
        ...
    path:
        full path to location.


    """

    module: Optional[str]
    version: Optional[str]
    kind: str
    path: str

    @classmethod
    def _deserialise(cls, *args, **kwargs):
        return cls(**kwargs)

    def __iter__(self):
        return iter([self.module, self.version, self.kind, self.path])


class Verbatim(Node):
    value: List[str]

    def __hash__(self):
        return hash(tuple(self.value))

    def __init__(self, value=None):
        self.value = value

    @classmethod
    def parse(cls, tokens):
        # assert False
        acc = []
        if len(tokens) < 5:
            return None
        if (tokens[0], tokens[1]) == ("`", "`") and tokens[2].strip():
            for i, t in enumerate(tokens[2:]):
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


class Link(Node):
    """
    Links are usually the end goal of a directive,
    they are a way to link to another document.
    They contain a text; which will be what the user will see,
    as well as a reference to the document pointed to.
    They should also have an attribute to know whether the link is a
     - Local item (same document)
     - Internal item (same module)
     - External item (another module)
     - Web : a url to another page non papyri aware.
     - Exist: bool wether the thing they point to exists.


    - I'm wondering if those should be descendant of directive not to lose information and be able to reconsruct the
    directive from it.
    - A Link might get several token for multiline; I'm not sure about that either, and wether the inner text should be
      a block or not.
    """

    value: str
    reference: RefInfo
    # kind likely should be deprecated, or renamed
    # either keep exists/true/false, but that can be a property as to wether reference is None ?
    kind: str
    exists: bool

    def __init__(self, value=None, reference=None, kind=None, exists=None):
        assert kind in ("exists", "missing", "local", "module", None), kind
        self.value = value
        self.reference = reference
        if reference is not None:
            assert isinstance(reference, RefInfo), f"{reference}, {value}"
        self.kind = kind
        self.exists = exists

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        return f"<Link: {self.value=} {self.reference=} {self.kind=} {self.exists=}>"


class Directive(Node):

    value: List[str]
    domain: Optional[str]
    role: Optional[str]

    def __hash__(self):
        return hash((tuple(self.value), self.domain, self.role))

    def __init__(self, value, domain, role):
        self.value = value
        self.domain = domain
        if domain:
            assert role
        self.role = role

    @classmethod
    def _instance(cls):
        return cls("", "", "")

    @property
    def text(self):
        return "".join(self.value)

    @classmethod
    def parse(cls, tokens):
        acc = []
        consume_start = None
        domain, role = None, None
        if (
            (len(tokens) > 2)
            and tokens[0] == "`"
            and tokens[1] != "`"
            and tokens[1].strip()
        ):
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

        if consume_start is None:
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
        return "<Directive " + prefix + "`" + "".join(self.value) + "`>"


class BlockMath(Node):
    value: str


class Math(Node):
    value: List[str]  # list of tokens not list of lines.

    @property
    def text(self):
        return "".join(self.value)

    def __hash__(self):
        assert False
        return hash(tuple(self.value))

    def _validate(self):
        pass
        # assert len(self.value) == 1, self.value
        # pass


class Word(Node):
    value: str

    @classmethod
    def _instance(cls):
        assert False
        return cls("")

    def __repr__(self):
        return UNDERLINE(self.value)

    def __len__(self):
        assert False
        return len(self.value)

    def __hash__(self):
        assert False
        return hash(self.value)


class Words(Node):
    """A sequence of words that does not start not ends with spaces"""

    value: str

    def __init__(self, value=None):
        self.value = value

    @classmethod
    def _instance(cls):
        return cls("")

    def __repr__(self):
        return UNDERLINE(self.value)

    def __len__(self):
        return len(self.value)

    def __hash__(self):
        return hash(self.value)


def lex(lines):
    acc = ""
    for i, l in enumerate(lines):
        assert isinstance(l, str), l
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
        if i != len(lines) - 1:
            yield " "


class _XList(Node):
    value: List[Union[Paragraph, EnumeratedList, BulletList, DefList, BlockQuote]]

    @property
    def children(self):
        return self.value

    @children.setter
    def children(self, children):
        self.value = children

    def __init__(self, value=None):
        self.value = value


class EnumeratedList(_XList):
    pass


class BulletList(_XList):
    pass


class FirstCombinator:
    def __init__(self, parsers):
        self.parsers = parsers

    def parse(self, tokens):
        for parser in self.parsers:
            res = parser.parse(tokens)
            if res is not None:
                return res

        return None


class Section(Node):
    children: List[
        Union[
            Code,
            Code2,
            Text,
            Fig,
            Paragraph,
            DefList,
            BlockDirective,
            BlockMath,
            Example,
            BlockVerbatim,
            Param,
            BulletList,
            EnumeratedList,
            BlockQuote,
            Admonition,
        ]
    ]
    # might need to be more complicated like verbatim.
    title: Optional[str]

    def __eq__(self, other):
        return super().__eq__(other)

    def __init__(self, children=None, title=None):
        if children is None:
            children = []
        self.children = children
        tt = get_type_hints(type(self))["children"].__args__[0].__args__
        for c in children:
            assert isinstance(c, tt), f"{c} not in {tt}"
        self.title = title

    def __getitem__(self, k):
        return self.children[k]

    def __setitem__(self, k, v):
        self.children[k] = v

    def __iter__(self):
        return iter(self.children)

    def append(self, item):
        self.children.append(item)

    def __repr__(self):
        rep = f"<{self.__class__.__name__} {self.title}:"
        for c in self.children:
            rep += "\n" + indent(repr(c).rstrip())
        rep += "\n>"
        return rep

    def empty(self):
        return len(self.children) == 0

    def __bool__(self):
        return len(self.children) >= 0

    def __len__(self):
        return len(self.children)


class Param(Node):
    param: str
    type_: str
    desc: List[
        Union[
            # Code,
            Text,
            Fig,
            Paragraph,
            DefListItem,
            DefList,
            BlockDirective,
            BlockMath,
            Example,
            BlockVerbatim,
            Admonition,
        ]
    ]

    def __init__(self, param=None, type_=None, desc=None):
        self.param = param
        self.type_ = type_
        self.desc = desc

    @property
    def children(self):
        return self.desc

    @children.setter
    def children(self, values):
        self.desc = values

    def __getitem__(self, index):
        return [self.param, self.type_, self.desc][index]

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: {self.param=}, {self.type_=}, {self.desc=}>"
        )

    def __hash__(self):
        assert False
        return hash((self.param, self.type_, self.desc))


class Token(Node):
    type: Optional[str]
    link: Union[Link, str]

    def __init__(self, link=None, type=None):
        self.link = link
        self.type = type

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.link=} {self.type=} >"


class Code2(Node):
    entries: List[Token]
    out: str
    ce_status: str

    def __init__(self, entries=None, out=None, ce_status=None):
        self.entries = entries
        self.out = out
        self.ce_status = ce_status


class Code(Node):
    entries: List[Tuple[Optional[str]]]
    out: str
    ce_status: str

    def __init__(self, entries=None, out=None, ce_status=None):
        self.entries = entries
        self.out = out
        self.ce_status = ce_status

    def _validate(self):
        for e in self.entries:
            pass
            # assert len(e) == 3

    @classmethod
    def _deserialise(cls, *args, **kwargs):
        inst = super()._deserialise(*args, **kwargs)
        for e in inst.entries:
            assert isinstance(e, tuple)
            for t in e:
                assert type(t) in (str, type(None)), inst.entries
        return inst


class Text(Node):
    value: str


class BlockQuote(Node):
    value: List[str]

    def __init__(self, value=None):
        self.value = value


class Fig(Node):
    value: str


def compress_word(stream):
    acc = []
    wds = ""
    assert isinstance(stream, list)
    for item in stream:
        if isinstance(item, Word):
            wds += item.value
        else:
            if wds:
                acc.append(Words(wds))
                wds = ""
            acc.append(item)
    if wds:
        acc.append(Words(wds))
    return acc


class Paragraph(Node):

    __slots__ = ["inner", "inline", "width"]

    inline: List[
        Union[
            Word,
            Words,
            Directive,
            Verbatim,
            Link,
            Math,
        ]
    ]

    inner: List[Union[Paragraph, BlockVerbatim, BulletList, EnumeratedList]]

    def __init__(self, inline, inner, width=80):
        for i in inline:
            assert isinstance(
                i,
                (
                    # Word,
                    Words,
                    Directive,
                    Verbatim,
                    Link,
                    Math,
                ),
            ), i
        self.inline = inline
        self.inner = inner
        self.width = width

    @property
    def children(self):
        return [*self.inline, *self.inner]

    @children.setter
    def children(self, new):
        inner = []
        inline = []
        for n in new:
            if isinstance(n, (Word, Words, Directive, Verbatim, Link, Math)):
                inline.append(n)
            else:
                break
        for n in new:
            if isinstance(n, (Paragraph, BlockVerbatim, BulletList, EnumeratedList)):
                inner.append(n)

        assert len(inner) + len(inline) == len(new)
        self.inner = inner
        self.inline = inline

    @classmethod
    def _instance(cls):
        return cls([], [])

    @classmethod
    def parse_lines(cls, lines):
        assert isinstance(lines, list), lines
        assert lines
        tokens = list(lex(lines))

        rest = tokens
        acc = []
        parser = FirstCombinator([Verbatim, Directive, Word])
        while rest:
            parsed, rest = parser.parse(rest)
            acc.append(parsed)

        return cls(compress_word(acc), [])

    def __repr__(self):

        rw = self.rewrap(self.children, self.width)

        p = "\n".join(["".join(repr(x) for x in line) for line in rw])
        return f"""<Paragraph:\n{p}>"""

    @classmethod
    def rewrap(cls, tokens, max_len):
        acc = [[]]
        clen = 0
        for t in tokens:
            try:
                lent = len(t)
            except TypeError:
                lent = 0
            if clen + lent > max_len:
                # remove whitespace at EOL
                try:
                    while acc and acc[-1][-1].is_whitespace():
                        acc[-1].pop()
                except IndexError:
                    pass
                acc.append([])
                clen = 0

            # do no append whitespace at SOL
            if clen == 0 and hasattr(t, "value") and t.is_whitespace():
                continue
            acc[-1].append(t)
            clen += lent
        # remove whitespace at EOF
        try:
            pass
            # while acc and acc[-1][-1].is_whitespace():
            #    acc[-1].pop()
        except IndexError:
            pass
        return acc

    def __hash__(self):
        return hash((tuple(self.children), self.width))

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.children == other.children)


def indent(text, marker="   |"):
    """
    Return the given text indented with 3 space plus a pipe for display.
    """
    lines = text.split("\n")
    return "\n".join(marker + l for l in lines)


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
        if ls := l.lstrip():
            yield (indent := len(l) - len(ls)), l
        else:
            yield indent, l


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
    I think the correct alternative is that each block may get an indented children,
    and that a block is thus:

        - a) The sequence of consecutive non blank lines with 0 indentation
        - b) The (potentially absent) blank lines leading to the indent block
        - c) The Raw indent block (we can decide to recurse, or not later)
        - d?) The trailing blank line at the end of the block leading to the next one.
                I think the blank line will be in the
        raw indent block

    """
    assert isinstance(lines, Lines)
    a: List[Lines]
    b: List[Lines]
    c: List[Lines]
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
        elif l.indent is None:
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


class Block(Node):
    """
    The following is wrong for some case, in particular if there are many paragraph in a row with 0 indent.
    we can't ignore blank lines.

    ---

    A chunk of lines that breaks when the indentation reaches::

        - the last of a list of blank lines if indentation is consistant
        - the last non-0  indented lines


    Note we likely want the _body_ lines and then the _indented_ lines if any, which would mean we
    cut after the first blank lines and expect indents, otherwise there is not indent.
    and likely if there is a blank lnes as  a property.

    ----

    I think the correct alternative is that each block may get an indented children, and that a block is thus::

        - 1) The sequence of consecutive non blank lines with 0 indentation
        - 2) The (potentially absent) blank lines leading to the indent block
        - 3) The Raw indent block (we can decide to recurse, or not later)
        - 4) The trailing blank line at the end of the block leading to the next one.

    """

    def __init__(self, lines, wh, ind, *, reason=None):
        if not lines:
            lines = [
                ":: workaround numpyoc summary/ext summary bug in take2::Line.__init__"
            ]
        self.lines = Lines(lines)
        self.wh = Lines(wh)
        self.ind = Lines(ind)
        self.reason = reason

    def __repr__(self):
        from typing import get_type_hints as gth

        attrs = gth(type(self)).keys()
        reprattr = ", ".join([f"{name}={getattr(self, name)}" for name in attrs])

        return f"<{self.__class__.__name__} '" + reprattr + "'>"


class BlockError(Block):
    @classmethod
    def from_block(cls, block):
        return cls(block.lines, block.wh, block.ind)


# class Section:
#    """
#    A section start (or not) with a header.
#
#    And have a body
#    """
#
#    def __init__(self, lines):
#        self.lines = lines
#
#    @property
#    def header(self):
#        if is_at_header(self.lines):
#            return self.lines[0:2]
#        else:
#            return None, None
#
#    @property
#    def body(self):
#        if is_at_header(self.lines):
#            return make_blocks_2(self.lines[2:])
#        else:
#            return make_blocks_2(self.lines)
#
#    def __repr__(self):
#        return (
#            f"<Section header='{self.header[0]}' body-len='{len(self.lines)}'> with\n"
#            + indent("\n".join([str(b) for b in self.body]) + "...END\n\n", "    |")
#        )


# wrapper around handling lines


class Line(Node):

    _line: str
    _number: int
    _offset: int

    def __init__(self, _line, _number, _offset=0):
        assert isinstance(_line, str)
        assert "\n" not in _line, _line
        self._line = _line
        self._number = _number
        self._offset = _offset

    def __eq__(self, other):
        for attr in ["_line", "_number", "_offset"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return type(self) == type(other)

    @classmethod
    def _instance(cls):
        return cls("", 0)

    @property
    def text(self):
        return self._line.rstrip()[self._offset :]

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


class Lines(Node):

    _lines: List[Line]

    def __init__(self, _lines=None):
        if _lines is None:
            _lines = []
        assert isinstance(_lines, (list, Lines))
        for l in _lines:
            assert isinstance(l, (str, Line)), f"got {l}"
            if isinstance(l, str):
                assert "\n" not in l
            if isinstance(l, Line):
                assert "\n" not in l._line

        self._lines = [
            l if isinstance(l, Line) else Line(l, n) for n, l in enumerate(_lines)
        ]

    def __eq__(self, other):
        return (type(self) == type(other)) and (self._lines == other._lines)

    @classmethod
    def _instance(cls):
        return cls([])

    def __iter__(self):
        return iter(self._lines)

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
        d = min([l.indent for l in self._lines if l.indent is not None])

        new_lines = []
        for l in self._lines:
            nl = Line(l._line, l._number, l._offset + d)
            new_lines.append(nl)
        return Lines(new_lines)

    def __len__(self):
        return len(self._lines)

    def __add__(self, other):
        if not isinstance(other, Lines):
            return NotImplemented
        return Lines(self._lines + other._lines)


# class Document:
#    def __init__(self, lines):
#        self.lines = lines
#
#    @property
#    def sections(self):
#        indices = header_lines(self.lines)
#        return [Section(l) for l in separate(self.lines, indices)]
#
#    def __repr__(self):
#        acc = ""
#        for i, s in enumerate(self.sections[0:]):
#            acc += "\n" + repr(s)
#        return "<Document > with" + indent(acc)


def is_at_header(lines) -> bool:
    """
    Given a list of lines (str), return wether or not we are (likely) at a header

    A header is (generally) of the form:
        - One line with text
        - one line some lenght with one of -_=*~ (and space on each side).

    In practice user do not use the same length, and some things may trigger
    false positive ( tracebacks in docstrings print with a long line of dashes
    (-).

    We could also peek at the line n-1, and make sure it is a blankline, also
    some libraries (scipy), use 0 level header with both over and underline
    (this is not implemented)

    We might also be able to find that headers are actually blocs as well, and
    blockify a full document, though we have to be careful, some thing so not
    have spaces after headers. (numpy.__doc__)
    """
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


class Header:
    """
    a header node
    """

    def __init__(self, lines):
        assert len(lines) >= 2, f"{lines=}"
        self._lines = lines
        self.level = None

    def __repr__(self):
        return (
            f"<Header {self.level}> with\n"
            + RED(indent(str(self._lines[0]), "    "))
            + "\n"
            + RED(indent(str(self._lines[1]), "    "))
            + "\n"
            + RED(indent("\n".join(str(x) for x in self._lines[2:]), "    "))
        )


class Admonition(Block):

    kind: str
    title: Optional[str]
    children: List[Paragraph]

    def __init__(self, kind=None, title=None, children=None):
        self.kind = kind
        self.children = children
        self.title = title


class BlockDirective(Block):
    lines: Lines
    wh: Lines
    ind: Lines

    directive_name: str
    args0: List[str]
    # TODO : this is likely wrong...
    inner: Optional[Paragraph]

    @property
    def children(self):
        if self.inner is not None:
            return [self.inner]
        else:
            return []

    @children.setter
    def children(self, value):
        assert len(value) in [0, 1]
        if len(value) == 0:
            assert not self.inner
        else:
            self.inner = value[0]

    def __init__(self, lines=None, wh=None, ind=None):
        if None in (lines, wh, ind):
            return
        self.lines = lines
        self.wh = wh
        self.ind = ind

        # numpy doc bug
        l = lines[0]._line
        if l.startswith("..version"):
            lines[0]._line = ".. " + l[2:]
        # end numpy doc bug
        # scipy bug....
        if lines[0].startswith("..Deprecated"):
            lines[0]._line = ".. deprecated:: 1.5.0"
        # end scipy bug.
        # assert lines[0].startswith(".. "), lines
        l0 = lines[0]
        pred, *postd = l0.split("::")
        # assert pred.startswith(".. ")
        self.directive_name = pred[3:].strip()
        if pred.startswith(".. |"):
            # TODO:
            print("replacement not implemented yet")
        elif " " in self.directive_name:
            assert False, repr(pred)
        self.args0 = postd
        if self.ind:
            self.inner = Paragraph.parse_lines([x.text for x in self.ind.dedented()])
        else:
            self.inner = None


class Comment(Block):
    lines: Lines
    wh: Lines
    ind: Lines

    def __init__(self, lines=None, wh=None, ind=None):
        if None in (lines, wh, ind):
            return
        self.lines = lines
        self.wh = wh
        self.ind = ind


class BlockVerbatim(Block):

    lines: Lines

    def __init__(self, lines):

        self.lines = lines

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.lines == other.lines)

    @classmethod
    def _instance(cls):
        return cls("")

    def __repr__(self):
        return f"<{self.__class__.__name__} '{len(self.lines)}'> with\n" + indent(
            "\n".join([str(l) for l in self.lines]), "    "
        )

    def to_json(self):
        return serialize(self, type(self))


class DefList(Block):
    children: List[DefListItem]

    def __init__(self, children=None):
        self.children = children

    def __repr__(self):
        return f"<{self.__class__.__name__} '{len(self.children)}'> with\n" + indent(
            "\n".join([str(l) for l in self.children]), "    "
        )


class DefListItem(Block):
    lines: Lines
    wh: Lines
    ind: Lines
    dt: Paragraph  # TODO: this is technically incorrect and should
    # be a single term, (word, directive or link is my guess).
    dd: List[Union[Paragraph, BulletList, EnumeratedList, BlockQuote, DefList]]

    @property
    def children(self):
        return [self.dt, *self.dd]

    @children.setter
    def children(self, value):
        self.dt, *self.dd = value

    def __init__(self, lines=None, wh=None, ind=None, dl=None, dd=None):
        self.lines = lines
        self.wh = wh
        self.ind = ind
        self.dt = dl
        assert isinstance(dd, (list, type(None))), dd
        self.dd = dd

    @classmethod
    def parse(cls, lines, wh, ind):
        dl = Paragraph.parse_lines([l.text.strip() for l in lines])
        assert len(dl.children) == 1
        dd = Paragraph.parse_lines([x.text for x in ind.dedented()])
        return cls(lines, wh, ind, dl, [dd])

    @classmethod
    def _deserialise(cls, **kwargs):
        inst = cls()
        for k, v in kwargs.items():
            setattr(inst, k, v)
        return inst


class Ref(Node):
    name: str
    ref: Optional[str]
    exists: Optional[bool]

    def __init__(self, name=None, ref=None, exists=None):
        self.name = name
        self.ref = ref
        self.exists = exists

    def __hash__(self):
        return hash((self.name, self.ref, self.exists))

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name} {self.ref} {self.exists}>"


class SeeAlsoItem(Node):
    name: Ref
    descriptions: List[Paragraph]
    # there are a few case when the lhs is `:func:something`... in scipy.
    type: Optional[str]

    def __init__(self, name=None, descriptions=None, type_=None):
        self.name = name
        if descriptions is not None:
            for d in descriptions:
                assert isinstance(d, Paragraph), repr(d)
        self.descriptions = descriptions
        self.type = type_

    # @classmethod
    # def from_json(cls, name, descriptions, type):
    #    assert isinstance(descriptions, list)
    #    return cls(Ref(**name), descriptions, type)
    #    assert isinstance(self.descriptions, list)

    def __hash__(self):
        return hash((self.name, tuple(self.descriptions)))

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: {self.name} {self.type} {self.descriptions}>"
        )


class Example(Block):
    lines: Lines
    wh: Lines
    ind: Lines

    def __init__(self, lines=None, wh=None, ind=None):
        self.lines = lines
        self.wh = wh
        self.ind = ind


def header_pass(block):
    """
    Check each block for potential header, if found, split (or extract) the given block into a header.

    Parameters
    ----------
    block : Block
        A block to split or extract

    Returns
    -------
    list
        A list of blocks and/or header

    """
    # TODO: Add logic to handle doctests raising exceptions.

    if len(block.lines) == 2 and is_at_header(block.lines):
        assert not block.ind
        return [Header(block.lines + block.wh)]
    elif len(block.lines) >= 2 and is_at_header(block.lines):
        h = Header(block.lines[:2])
        block.lines = block.lines[2:]
        return (h, block)
    return (block,)


def header_level_pass(blocks):
    """
    Iter over all top level nodes, updating the header level.

    For each encountered header, collect the type of marker use to underline,
    and increase the counter on newly encountered marker.

    Update the level attribute of the header with the corresponding value.

    Parameters
    ----------
    blocks : Block
        A block to split or extract

    """
    seen = {}
    for b in blocks:
        if not isinstance(b, Header):
            continue
        marker_set = set(b._lines[1].strip())
        assert len(marker_set) == 1, b
        marker = next(iter(marker_set))
        if marker not in seen:
            seen[marker] = len(seen)
        b.level = seen[marker]

    return blocks


def example_pass(block):
    if not type(block) == Block:
        return [block]
    if block.lines and block.lines[0].startswith(">>>"):
        return [Example(block.lines, block.wh, block.ind)]
    return [block]


def deflist_pass(blocks):
    acc = []
    deflist = []
    for block in blocks:
        if not type(block) == Block:
            acc.append(block)
            continue
        if len(block.lines) == 1:
            p = Paragraph.parse_lines([l.text.strip() for l in block.lines])
        if (
            len(block.lines) == 1
            and (not block.wh)
            and block.ind
            and len(p.children) == 1
        ):
            deflist.append(
                DefListItem.parse(
                    block.lines.dedented(), block.wh, block.ind.dedented()
                )
            )
        else:
            if deflist:
                acc.append(DefList(deflist))
                deflist = []
            acc.append(block)
    if deflist:
        acc.append(DefList(deflist))
    return acc


def deflist_item_pass(block):
    assert False
    if not type(block) == Block:
        return [block]
    if len(block.lines) == 1 and (not block.wh) and block.ind:
        return [DefListItem(block.lines, block.wh, block.ind)]
    return [block]


def block_directive_pass(block):
    if not type(block) == Block:
        return [block]
    if len(block.lines) >= 1 and (
        block.lines[0].startswith(".. ") and ("::" in block.lines[0].text)
    ):
        return [BlockDirective(block.lines, block.wh, block.ind)]
    return [block]


# TODO Comments
def block_comment(block):
    if not type(block) == Block:
        return [block]
    if len(block.lines) >= 1 and (block.lines[0].startswith(".. ")):
        assert "::" not in block.lines[0].text
        return [BlockDirective(block.lines, block.wh, block.ind)]
    return [block]


def paragraphs_pass(block):
    if not type(block) == Block:
        return [block]
    else:
        # likely incorrect for the indented part.
        if block.ind:
            assert isinstance(block.lines, Lines)
            lines = block.lines
            if not lines:
                return [BlockError.from_block(block)]
            if lines[-1]._line.endswith("::"):
                return [Paragraph.parse_lines([l._line for l in block.lines])] + [
                    BlockVerbatim(block.ind.dedented())
                ]
            else:
                sub = [Block(*b) for b in make_block_3(block.ind.dedented())]
                sub = deflist_pass(sub)
                sub = [x for pairs in sub for x in paragraphs_pass(pairs)]
                return [Paragraph.parse_lines([l._line for l in block.lines])] + sub
        else:
            return [Paragraph.parse_lines([l._line for l in block.lines])]


def empty_pass(doc):
    ret = []
    for b in doc:
        if not b.lines:
            assert not b.wh
            assert not b.ind
            continue
        ret.append(b)
    return ret


def get_object(qual):
    parts = qual.split(".")

    for i in range(len(parts), 1, -1):
        mod_p, _ = parts[:i], parts[i:]
        mod_n = ".".join(mod_p)
        try:
            __import__(mod_n)
            break
        except Exception:
            continue

    obj = __import__(parts[0])
    for p in parts[1:]:
        obj = getattr(obj, p)
    return obj


def assert_block_lines(blocks):
    for b in blocks:
        assert b.lines


def parse_rst_to_papyri_tree(text):
    """ """

    doc = [Block(*b) for b in make_block_3(Lines(text.split("\n"))[:])]
    assert_block_lines(doc), "raw blocks"
    doc = [x for pairs in doc for x in header_pass(pairs)]
    doc = header_level_pass(doc)
    doc = [x for pairs in doc for x in example_pass(pairs)]
    doc = [x for pairs in doc for x in block_directive_pass(pairs)]
    doc = deflist_pass(doc)
    doc = [x for pairs in doc for x in paragraphs_pass(pairs)]

    # TODO: third pass to set the header level for each header.
    # TODO: forth pass to make sections.

    #    print(b)
    # print(ex)
    # for w in [80, 120]:
    #    p = Paragraph.parse_lines(ex.split("\n"))
    #    p.width = w
    #    print(p)
    #    print()
    return doc


if __name__ == "__main__":
    if len(sys.argv) > 1:
        what = sys.argv[1]
    else:
        what = "numpy"
    ex = get_object(what).__doc__
    ex = dedent_but_first(ex)
    doc = main(ex)
    for b in doc:
        print(b)
