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
from typing import Any, List, NewType, Optional, Tuple, Union

import cbor2
from there import print

from .common_ast import Node, TAG_MAP, REV_TAG_MAP
from .miniserde import get_type_hints

from .utils import dedent_but_first

FullQual = NewType("FullQual", str)
Cannonical = NewType("Cannonical", str)


def register(value):
    assert value not in REV_TAG_MAP

    def _inner(type_):
        assert type_ not in TAG_MAP
        TAG_MAP[type_] = value
        REV_TAG_MAP[value] = type_

        return type_

    return _inner


register(tuple)(4444)


@register(4043)
class ExternalLink(Node):
    """
    ExternalLink are link to external resources.
    Most of the time they will be URL to other web resources,
    """

    value: str
    target: str


@register(4001)
class Verbatim(Node):
    value: List[str]

    def __eq__(self, other):
        if not type(self) == type(other):
            return False

        return self.text == other.text

    def __hash__(self):
        return hash(tuple(self.value))

    @property
    def text(self):
        return "".join(self.value)

    def __repr__(self):
        return "<Verbatim ``" + "".join(self.value) + "``>"


@register(4003)
class Directive(Node):
    value: str
    domain: Optional[str]
    role: Optional[str]

    def __hash__(self):
        return hash((tuple(self.value), self.domain, self.role))

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

     - Anchor: reference to a particular anchor in the target document.


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
    anchor: Optional[str] = None

    def __repr__(self):
        return f"<Link: {self.value=} {self.reference=} {self.kind=} {self.exists=}>"

    def __hash__(self):
        return hash((self.value, self.reference, self.kind, self.exists, self.anchor))


@register(4009)
class Strong(Node):
    content: Words

    @property
    def children(self):
        return [self.content]

    @children.setter
    def children(self, children):
        [self.content] = children

    def __hash__(self):
        return hash(repr(self))


class Leaf(Node):
    value: str


@register(4005)
class Math(Leaf):
    pass


from .myst_ast import MText, MParagraph, MEmphasis, MInlineCode, MCode


class IntermediateNode(Node):
    """
    This is just a dummy class for Intermediate node that should not make it to the final Product
    """

    pass


@register(4004)
class BlockMath(Leaf):
    pass


@register(4027)
class SubstitutionDef(Node):
    name: str
    directive: BlockDirective


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


@register(4024)
class Fig(Node):
    value: RefInfo


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
        assert isinstance(self.path, str)
        return iter([self.module, self.version, self.kind, self.path])


class Word(IntermediateNode):
    """
    This is a temporary node, while we visit the tree-sitter tree,
    we will compress those into words with subsequent whitespace


    """

    value: str


@register(4007)
class Words(Node):
    """A sequence of words that does not start not ends with spaces"""

    value: str

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


class _XList(Node):
    children: List[ListItem]


@register(4006)
class ListItem(Node):
    children: List[
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
            Comment,
            MParagraph,
            MCode,
        ]
    ]


@register(4039)
class EnumeratedList(_XList):
    pass


@register(4040)
class BulletList(_XList):
    pass


@register(4011)
class Signature(Node):
    value: Optional[str]


@register(4012)
class NumpydocExample(Node):
    value: List[str]
    title = "Examples"


@register(4013)
class NumpydocSeeAlso(Node):
    value: List[SeeAlsoItem]
    title = "See Also"


@register(4014)
class NumpydocSignature(Node):
    value: str
    title = "Signature"


@register(4015)
class Section(Node):
    children: List[
        Union[
            Transition,
            # Code,
            MCode,
            Code2,
            Code3,
            Unimplemented,
            Comment,
            Target,
            Words,
            Fig,
            Options,
            Paragraph,
            MParagraph,
            DefList,
            BlockDirective,
            Unimplemented,
            BlockMath,
            BlockVerbatim,
            Parameters,
            BulletList,
            EnumeratedList,
            BlockQuote,
            Admonition,
            FieldList,
            Target,
            SubstitutionRef,
            SubstitutionDef,
        ]
    ]
    # might need to be more complicated like verbatim.
    title: Optional[str]
    level: int = 0
    target: Optional[str] = None

    def __eq__(self, other):
        return super().__eq__(other)

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


@register(4026)
class Parameters(Node):
    children: List[Param]

    def validate(self):
        assert len(self.children) > 0
        return super().validate()


@register(4016)
class Param(Node):
    param: str
    type_: str
    desc: List[
        Union[
            # Code,
            Words,
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
            MParagraph,
            MCode,
        ]
    ]

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
    """
    A single token in a code block.

    Paramters
    ---------
    type : str, optional
        this currently is a classname use by pygments for highlighting.
    link : str | Link(value, reference, kind, exists)
        this is either a string (the value to display), or a link that point to a given page.

    """

    link: Union[Link, str]
    type: Optional[str]

    @property
    def children(self):
        return [self.link, self.type]

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.link=} {self.type=} >"


@register(4018)
class Unimplemented(Node):
    placeholder: str
    value: str

    def __repr__(self):
        return f"<Unimplemented {self.placeholder!r} {self.value!r}>"


@register(4029)
class Code3(Node):
    """
    Trying to think about the code entries,
    after trying to render a few of them, I think we need to change the structure a bit.
    Mostly I think we need to

     - store each line independently,
     - possibly each line should/may get a "prompt" indicator (which should be non-selectable in the end),
       or should the prompt be at the code level ? with first prompt continuation prompt ?
       Mostly this is because code might be python, or bash, or anything else.
     - the fact that we have multiple lines, means that we can highlight some of the lines which is common  but hard in
       code blocks.
     - it also looks like the rendering is often hard if we have to treat new lines separately.
     - "prompt" can also serve in the margin to show the lien numbers in a file.
    """

    status: str
    children: List[CodeLine]
    out: str


@register(4044)
class CodeLine(Node):
    prompt: str
    entries: List[Token]
    highlighted: bool


@register(4020)
class Code2(Node):
    entries: List[Token]
    out: str
    ce_status: str

    @property
    def children(self):
        return [*self.entries, self.out, self.ce_status]

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.entries=} {self.out=} {self.ce_status=}>"


class GenToken(Node):
    value: str
    qa: Optional[str]
    pygmentclass: str


class Code(Node):
    entries: List[GenToken]
    out: str
    ce_status: str

    def validate(self):
        for x in self.entries:
            assert isinstance(x, GenToken)

        return super().validate()

    def _validate(self):
        for e in self.entries:  # noqa: B007
            pass
            # assert len(e) == 3

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.entries=} {self.out=} {self.ce_status=}>"


@register(4023)
class BlockQuote(Node):
    children: List[
        Union[
            Paragraph,
            BlockVerbatim,
            BulletList,
            DefList,
            EnumeratedList,
            BlockDirective,
            BlockQuote,
            FieldList,
            Admonition,
            Unimplemented,
            Comment,
            BlockMath,
            MParagraph,
            MCode,
        ]
    ]


def compress_word(stream) -> List[Any]:
    acc = []
    wds = ""
    assert isinstance(stream, list)
    for item in stream:
        if isinstance(item, MText):
            wds += item.value
        else:
            if type(item).__name__ == "Whitespace":
                acc.append(MText(item.value))
                wds = ""
            else:
                if wds:
                    acc.append(MText(wds))
                    wds = ""
                acc.append(item)
    if wds:
        acc.append(MText(wds))
    return acc


@register(4019)
class Transition(Node):
    pass


inline_nodes = tuple(
    [
        Words,
        Strong,
        Emph,
        Directive,
        Verbatim,
        Link,
        ExternalLink,
        Math,
        SubstitutionRef,
    ]
)


@register(4025)
class Paragraph(Node):
    children: List[
        Union[
            Words,
            MText,
            MCode,
            Strong,
            Unimplemented,
            Emph,
            MEmphasis,
            MInlineCode,
            Target,
            Directive,
            Verbatim,
            Link,
            ExternalLink,
            Math,
            SubstitutionRef,
        ]
    ]

    def __hash__(self):
        return hash((tuple(self.children)))

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.children == other.children)


@register(4038)
class Admonition(Node):
    kind: str
    title: Optional[str]
    children: List[
        Union[
            Paragraph,
            MParagraph,
            MCode,
            BulletList,
            BlockVerbatim,
            BlockQuote,
            DefList,
            # I dont' like nested block directive/Admonitions.
            BlockDirective,
            Admonition,
            Unimplemented,  # skimage.util._regular_grid.regular_grid
            EnumeratedList,
        ]
    ]


@register(4021)
class TocTree(Node):
    children: List[TocTree]
    title: str
    ref: RefInfo
    open: bool = False
    current: bool = False


@register(4031)
class BlockDirective(Node):
    name: str
    argument: str
    options: List[Tuple[str]]
    content: str

    def validate(self):
        assert isinstance(self.name, str)
        assert isinstance(self.argument, str)
        assert isinstance(self.options, list)
        for it in self.options:
            assert isinstance(it, tuple)
            k, v = it
            assert isinstance(k, str)
            assert isinstance(v, str)
        assert isinstance(self.content, str)
        return super().validate()

    def _post_deserialise(self):
        self.options = [tuple(x) for x in self.options]

    @property
    def value(self):
        return [self.name, self.argument, self.options, self.content]


@register(4032)
class BlockVerbatim(Node):
    value: str

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.value == other.value)

    def __repr__(self):
        return f"<{self.__class__.__name__} '{len(self.value)}'>"


@register(4033)
class DefList(Node):
    children: List[DefListItem]


@register(4034)
class Options(Node):
    values: List[str]


@register(4035)
class FieldList(Node):
    children: List[FieldListItem]


@register(4036)
class FieldListItem(Node):
    name: List[
        Union[
            Paragraph,
            Words,
            Verbatim,
            MText,
            MCode,
        ]
    ]
    body: List[
        Union[
            Words,
            Paragraph,
            Verbatim,
            Admonition,
            BlockDirective,
            BulletList,
            MText,
            MParagraph,
            MCode,
        ]
    ]

    def validate(self):
        for p in self.body:
            assert isinstance(p, Paragraph), p
        if self.name:
            assert len(self.name) == 1, (self.name, [type(n) for n in self.name])
        return super().validate()

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
    dt: Union[Paragraph, MParagraph]  # TODO: this is technically incorrect and should
    # be a single term, (word, directive or link is my guess).
    dd: List[
        Union[
            Paragraph,
            MParagraph,
            BulletList,
            EnumeratedList,
            BlockQuote,
            DefList,
            BlockDirective,
            Unimplemented,
            Admonition,
            BlockMath,
            BlockVerbatim,
            Optional[TocTree],  # remove this, that should not be the case ?
        ]
    ]

    @property
    def children(self):
        return [self.dt, *self.dd]

    @children.setter
    def children(self, value):
        self.dt, *self.dd = value
        self.validate()


@register(4028)
class SeeAlsoItem(Node):
    name: Link
    descriptions: List[Union[Paragraph, MParagraph]]
    # there are a few case when the lhs is `:func:something`... in scipy.
    type: Optional[str]

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

    def _available_tags(self):
        k = self._rev_map.keys()
        mi, ma = min(k), max(k)
        return set(range(mi, ma + 2)) - set(k)


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
