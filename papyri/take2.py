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


import typing
from typing import List

from papyri.miniserde import deserialize, get_type_hints, serialize


def not_type_check(item, annotation):

    if not hasattr(annotation, "__origin__"):
        if isinstance(item, annotation):
            return None
        else:
            return f"expecting {annotation} got {type(item)}"
    elif annotation.__origin__ is dict:
        if not isinstance(item, dict):
            return f"got  {type(item)}, Yexpecting list"
        inner_type = annotation.__args__[0]
        a = [not_type_check(i, inner_type) for i in item.keys()]
        ax = [x for x in a if x is not None]
        inner_type = annotation.__args__[1]
        b = [not_type_check(i, inner_type) for i in item.values()]
        bx = [x for x in b if x is not None]
        if ax:
            return ":invalid key type {ax[0]}"
        if bx:
            return bx[0]
        return None
    elif annotation.__origin__ in (list, tuple):
        # technically incorrect
        if not isinstance(item, (list, tuple)):
            return f"got  {type(item)}, Yexpecting list"
        assert len(annotation.__args__) == 1
        inner_type = annotation.__args__[0]

        b = [not_type_check(i, inner_type) for i in item]

        bp = [x for x in b if x is not None]
        if bp:
            return bp[0]
        else:
            return None
    elif annotation.__origin__ is typing.Union:
        if any([not_type_check(item, arg) is None for arg in annotation.__args__]):
            return None
        return f"expecting one of {annotation!r}, got {item!r}"
    raise ValueError(item, annotation)


def _invalidate(obj, depth=0):
    """
    Recursively validate type anotated classes.
    """

    annotations = get_type_hints(type(obj))
    for k, v in annotations.items():
        item = getattr(obj, k)
        res = not_type_check(item, v)
        if res:
            return f"{k} field of  {type(obj)} : {res}"

        if isinstance(item, (list, tuple)):
            for ii, i in enumerate(item):
                sub = _invalidate(i, depth + 1)
                if sub is not None:
                    return f"{k}.{ii}." + sub
        if isinstance(item, dict):
            for ii, i in item.items():
                sub = _invalidate(i, depth + 1)
                if sub is not None:
                    return f"{k}.{ii}." + sub
        else:
            sub = _invalidate(item, depth + 1)
            if sub is not None:
                return f"{k}.{ii}." + sub

    # return outcome,s


def validate(obj):
    res = _invalidate(obj)
    if res:
        raise ValueError(f"Wrong type at field :: {res}")


class Base:
    def validate(self):
        validate(self)
        return self

    @classmethod
    def _instance(cls):
        return cls()

    @classmethod
    def _deserialise(cls, **kwargs):
        # print("will deserialise", cls)
        return cls(**kwargs)


class Node(Base):
    def __init__(self, *args):
        tt = get_type_hints(type(self))
        for attr, val in zip(tt, args):
            setattr(self, attr, val)

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

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if not type(self) == type(other):
            return False

        return self.text == other.text

    def __hash__(self):
        return hash(tuple(self.value))

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

    value: str
    domain: Optional[str]
    role: Optional[str]

    def __hash__(self):
        return hash((tuple(self.value), self.domain, self.role))

    def __init__(self, value, domain, role):
        assert isinstance(value, str)
        assert "`" not in value, value
        self.value = value
        self.domain = domain
        if domain is not None:
            assert isinstance(domain, str), domain
        if domain:
            assert role
        self.role = role
        if role is not None:
            assert isinstance(role, str), role
            assert ":" not in role

    @classmethod
    def _instance(cls):
        return cls("", "", "")

    def __eq__(self, other):
        return (
            (type(self) == type(other))
            and (self.role == other.role)
            and (other.domain == self.domain)
            and (self.value == other.value)
        )

    def __len__(self):
        return len(self.value) + len(self.prefix) + 2

    @property
    def prefix(self):
        prefix = ""
        if self.domain:
            prefix += ":" + self.domain
        if self.role:
            prefix += ":" + self.role + ":"
        return prefix

    def __repr__(self):
        return f"<Directive {self.prefix}`{self.value}`>"


class BlockMath(Node):
    value: str

    def __init__(self, value):
        self.value = value


class Math(Node):
    value: List[str]  # list of tokens not list of lines.

    def __init__(self, value):
        assert isinstance(value, list)
        self.value = value

    @property
    def text(self):
        return "".join(self.value)

    def __hash__(self):
        return hash(tuple(self.value))

    def _validate(self):
        pass
        # assert len(self.value) == 1, self.value
        # pass


class Word(Node):
    value: str

    def __init__(self, value):
        self.value = value

    @classmethod
    def _instance(cls):
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

    def __init__(self, value):
        self.value = value

    @classmethod
    def _instance(cls):
        return cls("")

    def __eq__(self, other):
        return type(self) == type(other) and self.value.strip() == other.value.strip()

    def __repr__(self):
        return UNDERLINE(self.value)

    def __len__(self):
        return len(self.value)

    def __hash__(self):
        return hash(self.value)


class Emph(Node):
    value: Words

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(repr(self))

    @property
    def children(self):
        return [self.value]

    @children.setter
    def children(self, children):
        [self.value] = children

    def __repr__(self):
        return "*" + repr(self.value) + "*"


class Strong(Node):
    content: Words

    def __init__(self, content):
        self.content = content

    @property
    def children(self):
        return [self.content]

    @children.setter
    def children(self, children):
        [self.content] = children

    def __repr__(self):
        return "**" + repr(self.content) + "**"

    def __hash__(self):
        return hash(repr(self))

    def is_whitespace(self):
        return False


class _XList(Node):
    value: List[
        Union[
            Paragraph,
            EnumeratedList,
            BulletList,
            Target,
            DefList,
            BlockQuote,
            BlockVerbatim,
            BlockDirective,
            BlockMath,
            Unimplemented,
            Admonition,
        ]
    ]

    @property
    def children(self):
        return self.value

    @children.setter
    def children(self, children):
        self.value = children

    def __init__(self, value):
        self.value = value


class EnumeratedList(_XList):
    pass


class BulletList(_XList):
    pass


class Signature(Node):
    value: Optional[str]

    def __init__(self, value):
        self.value = value


class NumpydocExample(Node):
    value: List[str]

    def __init__(self, value):
        self.title = "Examples"
        self.value = value


class NumpydocSeeAlso(Node):
    value: List[SeeAlsoItem]

    def __init__(self, value):
        self.title = "See Also"
        self.value = value


class NumpydocSignature(Node):
    value: str

    def __init__(self, value):
        self.title = "Signature"
        self.value = value


class Section(Node):
    children: List[
        Union[
            Code,
            Code2,
            Unimplemented,
            Comment,
            Target,
            Text,
            Fig,
            Options,
            Paragraph,
            DefList,
            BlockDirective,
            Unimplemented,
            BlockMath,
            BlockVerbatim,
            Param,
            BulletList,
            EnumeratedList,
            BlockQuote,
            Admonition,
            FieldList,
            Target,
            SubstitutionRef,
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
        if title == "See also":
            title = "See Also"
        if title == "Arguments":
            title = "Parameters"
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
            BlockVerbatim,
            Admonition,
            BulletList,
            BlockQuote,
            EnumeratedList,
        ]
    ]

    def __init__(self, param, type_, desc):
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

    def __init__(self, link, type):
        self.link = link
        self.type = type

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.link=} {self.type=} >"


class Unimplemented(Node):
    value: str
    placeholder: str

    def __init__(self, placeholder, value):
        self.placeholder = placeholder
        self.value = value

    def __repr__(self):
        return f"<Unimplemented {self.placeholder!r} {self.value!r}>"


class _Dummy(Node):
    value: str

    def __init__(self, value):
        self.value = value


class SubstitutionRef(_Dummy):
    pass


class Target(_Dummy):
    pass


class Code2(Node):
    entries: List[Token]
    out: str
    ce_status: str

    def __init__(self, entries, out, ce_status):
        self.entries = entries
        self.out = out
        self.ce_status = ce_status

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.entries=} {self.out=} {self.ce_status=}>"


class Code(Node):
    entries: List[Tuple[Optional[str]]]
    out: str
    ce_status: str

    def __init__(self, entries, out, ce_status):
        self.entries = entries
        self.out = out
        self.ce_status = ce_status

    def _validate(self):
        for e in self.entries:  # noqa: B007
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

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.entries=} {self.out=} {self.ce_status=}>"


class Text(Node):
    value: str

    def __init__(self, value):
        self.value = value


class BlockQuote(Node):
    value: List[str]

    def __init__(self, value):
        self.value = value


class Fig(Node):
    value: str

    def __init__(self, value):
        self.value = value


def compress_word(stream):
    acc = []
    wds = ""
    assert isinstance(stream, list)
    for item in stream:
        if isinstance(item, Word):
            wds += item.value
        else:
            if type(item).__name__ == "Whitespace":
                acc.append(Words(item.value))
                wds = ""
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
            Strong,
            Unimplemented,
            Emph,
            Target,
            Directive,
            Verbatim,
            Link,
            Math,
            SubstitutionRef,
        ]
    ]

    inner: List[
        Union[Paragraph, BlockVerbatim, BulletList, EnumeratedList, Unimplemented]
    ]

    def __init__(self, inline, inner, width=80):
        for i in inline:
            assert isinstance(
                i,
                (
                    # Word,
                    Strong,
                    Emph,
                    Words,
                    Unimplemented,
                    Directive,
                    Verbatim,
                    Link,
                    Math,
                    SubstitutionRef,
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
            if isinstance(
                n,
                (
                    Word,
                    Words,
                    Directive,
                    Verbatim,
                    Link,
                    Math,
                    Strong,
                    Emph,
                    SubstitutionRef,
                    Unimplemented,
                ),
            ):
                inline.append(n)
            else:
                break
        for n in new:
            if isinstance(n, (Paragraph, BlockVerbatim, BulletList, EnumeratedList)):
                inner.append(n)

        assert len(inner) + len(inline) == len(new), (inner, inline, new)

        self.inner = inner
        self.inline = inline

    @classmethod
    def _instance(cls):
        return cls([], [])

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

    def __repr__(self):
        from typing import get_type_hints as gth

        attrs = gth(type(self)).keys()
        reprattr = ", ".join([f"{name}={getattr(self, name)}" for name in attrs])

        return f"<{self.__class__.__name__} '" + reprattr + "'>"


class BlockError(Block):
    @classmethod
    def from_block(cls, block):
        return cls(block.lines, block.wh, block.ind)


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


class Comment(Block):

    value: str

    def __init__(self, value):
        """
        Comment should not make it in the final document,
        but we store them for now, to help with error reporting and
        custom transformations.
        """
        self.value = value


class BlockDirective(Block):

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

    def __init__(self, *, directive_name, args0, inner):
        self.directive_name = directive_name
        self.args0 = args0
        self.inner = inner


class BlockVerbatim(Block):

    value: str

    def __init__(self, value):

        assert isinstance(value, str)
        self.value = value

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.value == other.value)

    @classmethod
    def _instance(cls):
        return cls("")

    def __repr__(self):
        return f"<{self.__class__.__name__} '{len(self.value)}'>"

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


class Options(Block):

    values: List[str]

    def __init__(self, values):
        self.values = values


class FieldList(Block):
    children: List[FieldListItem]

    def __init__(self, children=None):
        self.children = children

    def __repr__(self):
        return f"<{self.__class__.__name__} '{len(self.children)}'> with\n" + indent(
            "\n".join([str(l) for l in self.children]), "    "
        )


class FieldListItem(Block):
    name: List[Union[Paragraph, Word, Words]]
    body: List[Union[Words, Paragraph, Word]]

    def __init__(self, name=None, body=None):
        if body is None:
            body = []
        for p in body:
            assert isinstance(p, Paragraph), p
        if name:
            assert len(name) == 1, (name, [type(n) for n in name])
        self.name = name
        self.body = body

    @property
    def children(self):
        if isinstance(self.name, Word):
            return [self.name, *self.body]
        else:
            return [*self.name, *self.body]

    @children.setter
    def children(self, value):
        x, *y = value
        self.name = [x]
        self.body = y


class DefListItem(Block):
    dt: Paragraph  # TODO: this is technically incorrect and should
    # be a single term, (word, directive or link is my guess).
    dd: List[
        Union[
            Paragraph,
            BulletList,
            EnumeratedList,
            BlockQuote,
            DefList,
            BlockDirective,
            Unimplemented,
            Admonition,
            BlockMath,
            BlockVerbatim,
        ]
    ]

    @property
    def children(self):
        return [self.dt, *self.dd]

    @children.setter
    def children(self, value):
        self.dt, *self.dd = value

    def __init__(self, dt=None, dd=None):
        self.dt = dt
        assert isinstance(dd, (list, type(None))), dd
        self.dd = dd

    @classmethod
    def _deserialise(cls, **kwargs):
        inst = cls(**kwargs)
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

    def __init__(self, name=None, descriptions=None, type=None):
        self.name = name
        if descriptions is not None:
            for d in descriptions:
                assert isinstance(d, Paragraph), repr(d)
        self.descriptions = descriptions
        self.type = type

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


def parse_rst_section(text):
    """
    This should at some point be completely replaced by tree sitter.
    in particular `from ts import parse`
    """

    from .ts import parse

    items = parse(text.encode())
    if len(items) == 0:
        return []
    if len(items) == 1:
        [section] = items
        return section.children
    raise ValueError("Multiple sections present")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        what = sys.argv[1]
    else:
        what = "numpy"
    ex = get_object(what).__doc__
    ex = dedent_but_first(ex)
    doc = parse_rst_section(ex)
    for b in doc:
        print(b)
