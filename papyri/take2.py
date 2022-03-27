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
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, NewType

import cbor2

from papyri.miniserde import deserialize, get_type_hints, serialize
from papyri.utils import dedent_but_first

FullQual = NewType("FullQual", str)
Cannonical = NewType("Cannonical", str)


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
                return f"{k}.{sub}." + sub

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


TAG_MAP: Dict[Any, int] = {}
REV_TAG_MAP: Dict[int, Any] = {}


def register(value):
    assert value not in REV_TAG_MAP

    def _inner(type_):
        assert type_ not in TAG_MAP
        TAG_MAP[type_] = value
        REV_TAG_MAP[value] = type_

        return type_

    return _inner


class Node(Base):
    def __init__(self, *args):
        tt = get_type_hints(type(self))
        for attr, val in zip(tt, args):
            setattr(self, attr, val)

    def cbor(self, encoder):

        tag = TAG_MAP[type(self)]
        attrs = get_type_hints(type(self))
        encoder.encode(cbor2.CBORTag(tag, [getattr(self, k) for k in attrs]))

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
        tt = get_type_hints(type(self))
        acc = ""
        for t in tt:
            acc += f"{t}: {getattr(self, t)!r}\n"

        return f"<{self.__class__.__name__}: \n{indent(acc)}>"

    def to_json(self):
        return serialize(self, type(self))

    @classmethod
    def from_json(cls, data):
        return deserialize(cls, cls, data)


class Leaf(Node):
    value: str

    def __init__(self, value):
        self.value = value


class IntermediateNode(Node):
    """
    This is just a dummy class for Intermediate node that should not make it to the final Product
    """

    pass


@register(4004)
class BlockMath(Leaf):
    pass


@register(4041)
class SubstitutionRef(Leaf):
    pass


@register(4042)
class Target(Leaf):
    pass


@register(4030)
class Comment(Leaf):
    """
    Comment should not make it in the final document,
    but we store them for now, to help with error reporting and
    custom transformations.
    """


@register(4022)
class Text(Leaf):
    pass


@register(4024)
class Fig(Leaf):
    pass


@register(4000)
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

    def __iter__(self):
        return iter([self.module, self.version, self.kind, self.path])


@register(4001)
class Verbatim(Node):
    value: List[str]

    def __init__(self, value):
        assert isinstance(value, list)
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
        return "``" + "".join(self.value) + "``"


@register(4043)
class ExternalLink(Node):
    """
    ExternalLink are link to external resources.
    Most of the time they will be URL to other web resources,
    """

    value: str
    target: str

    def __init__(self, value, target):
        self.value = value
        self.target = target


@register(4002)
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

    @property
    def children(self):
        return [self.value, self.reference, self.kind, self.exists]

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        return f"<Link: {self.value=} {self.reference=} {self.kind=} {self.exists=}>"


@register(4003)
class Directive(Node):

    value: str
    domain: Optional[str]
    role: Optional[str]

    def __hash__(self):
        return hash((tuple(self.value), self.domain, self.role))

    def __init__(self, value, domain, role):
        # if value == "NpyIter_MultiNew":
        #    breakpoint()
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


@register(4005)
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


class Word(IntermediateNode):
    """
    This is a temporary node, while we visit the tree-sitter tree,
    we will compress those into words with subsequent whitespace


    """

    value: str

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value


@register(4007)
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
        return self.value

    def __len__(self):
        return len(self.value)

    def __hash__(self):
        return hash(self.value)


@register(4008)
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


@register(4009)
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


@register(4039)
class EnumeratedList(_XList):
    pass


@register(4040)
class BulletList(_XList):
    pass


@register(4011)
class Signature(Node):
    value: Optional[str]

    def __init__(self, value):
        self.value = value


@register(4012)
class NumpydocExample(Node):
    value: List[str]

    def __init__(self, value):
        self.title = "Examples"
        self.value = value


@register(4013)
class NumpydocSeeAlso(Node):
    value: List[SeeAlsoItem]

    def __init__(self, value):
        self.title = "See Also"
        self.value = value


@register(4014)
class NumpydocSignature(Node):
    value: str

    def __init__(self, value):
        self.value = value
        self.title = "Signature"


@register(4015)
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

    def empty(self):
        return len(self.children) == 0

    def __bool__(self):
        return len(self.children) >= 0

    def __len__(self):
        return len(self.children)


@register(4016)
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


@register(4017)
class Token(Node):
    type: Optional[str]
    link: Union[Link, str]

    def __init__(self, link, type):
        self.link = link
        self.type = type

    @property
    def children(self):
        return [self.link, self.type]

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.link=} {self.type=} >"


@register(4018)
class Unimplemented(Node):
    value: str
    placeholder: str

    def __init__(self, placeholder, value):
        self.placeholder = placeholder
        self.value = value

    def __repr__(self):
        return f"<Unimplemented {self.placeholder!r} {self.value!r}>"


@register(4020)
class Code2(Node):
    entries: List[Token]
    out: str
    ce_status: str

    def __init__(self, entries, out, ce_status):
        self.entries = entries
        self.out = out
        self.ce_status = ce_status

    @property
    def children(self):
        return [*self.entries, self.out, self.ce_status]

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.entries=} {self.out=} {self.ce_status=}>"


@register(4021)
class Code(Node):
    entries: List[Tuple[Optional[str]]]
    out: str
    ce_status: str

    def __init__(self, entries, out: str, ce_status):
        self.entries = entries
        self.out = out
        self.ce_status = ce_status

    def _validate(self):
        for e in self.entries:  # noqa: B007
            pass
            # assert len(e) == 3

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.entries=} {self.out=} {self.ce_status=}>"


@register(4023)
class BlockQuote(Node):
    value: List[str]

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


@register(4025)
class Paragraph(Node):

    __slots__ = ["inner", "inline", "width"]

    inline: List[
        Union[
            Words,
            Strong,
            Unimplemented,
            Emph,
            Target,
            Directive,
            Verbatim,
            Link,
            ExternalLink,
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
                    Strong,
                    Emph,
                    Words,
                    Unimplemented,
                    Directive,
                    Verbatim,
                    Link,
                    ExternalLink,
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
                    Words,
                    Directive,
                    Verbatim,
                    Link,
                    ExternalLink,
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


@register(4038)
class Admonition(Node):

    kind: str
    title: Optional[str]
    children: List[Paragraph]

    def __init__(self, kind=None, title=None, children=None):
        self.kind = kind
        self.children = children
        self.title = title


@register(4031)
class BlockDirective(Node):

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

    def __init__(self, directive_name, args0, inner):
        self.directive_name = directive_name
        self.args0 = args0
        self.inner = inner


@register(4032)
class BlockVerbatim(Node):

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


@register(4033)
class DefList(Node):
    children: List[DefListItem]

    def __init__(self, children=None):
        self.children = children


@register(4034)
class Options(Node):

    values: List[str]

    def __init__(self, values):
        self.values = values


@register(4035)
class FieldList(Node):
    children: List[FieldListItem]

    def __init__(self, children=None):
        self.children = children


@register(4036)
class FieldListItem(Node):
    name: List[
        Union[
            Paragraph,
            Words,
        ]
    ]
    body: List[
        Union[
            Words,
            Paragraph,
            # Word
        ]
    ]

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
        return [*self.name, *self.body]

    @children.setter
    def children(self, value):
        x, *y = value
        self.name = [x]
        self.body = y


@register(4037)
class DefListItem(Node):
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


@register(4027)
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

    @property
    def children(self):
        return [self.name, self.ref, self.exists]


@register(4028)
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

    @property
    def children(self):
        return [self.name, self.type, *self.descriptions]

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


class Encoder:
    def __init__(self, rev_map):
        self._rev_map = rev_map

    def encode(self, obj):
        return cbor2.dumps(obj, default=lambda encoder, obj: obj.cbor(encoder))

    def _type_from_tag(self, tag):
        return self._rev_map[tag.tag]

    def _tag_hook(self, decoder, tag, shareable_index=None):
        type_ = self._type_from_tag(tag)

        tt = get_type_hints(type_)
        kwds = {k: t for k, t in zip(tt, tag.value)}
        return type_(**kwds)

    def decode(self, bytes):
        return cbor2.loads(bytes, tag_hook=self._tag_hook)


encoder = Encoder(REV_TAG_MAP)


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
