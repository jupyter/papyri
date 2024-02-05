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
from typing import Any, List, Optional, Union

import cbor2
from there import print

from .common_ast import Node, REV_TAG_MAP, register, UnserializableNode
from .miniserde import get_type_hints

from .utils import dedent_but_first

import logging

log = logging.getLogger(__file__)


register(tuple)(4444)


@register(4003)
class Directive(Node):
    value: str
    domain: Optional[str]
    role: Optional[str]

    def __init__(self, value, domain, role):
        assert "\n" not in value, f"Directive should not contain newline {value}"
        super().__init__(value, domain, role)

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
        return f"<Directive {self.prefix}`{self.value}` `{self.to_dict()}`>"

    def __str__(self):
        assert False


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

    - In general link won't end up in the final Json that is rendered as they will need to be resolved at runtime ?
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


class Leaf(Node):
    value: str


class SubstitutionDef(UnserializableNode):
    """
    A Substitution Definition should be in the form of

    .. raw:: rst

        .. |value| inline_directive:: text to be inserted

    Currently, the inline_directive can only be ``replace`` or ``image``. In the
    future, we want to support any inline directives, including custom
    user-defined directives.
    """

    value: str
    children: List[Union[ReplaceNode, MImage]]

    def __init__(self, value, children):
        self.value = value
        assert isinstance(children, list)
        assert len(children) == 1
        assert isinstance(children[0], UnprocessedDirective)

        if children[0].name == "image":
            assert len(children) == 1
            self.children = [MImage(url=children[0].args, alt="")]
        elif children[0].name == "replace":
            assert len(children) == 1
            self.children = [
                ReplaceNode(value=self.value, text=children[0].args, children=children)
            ]
        else:
            self.children = [
                ReplaceNode(value=self.value, text=children[0].args, children=children)
            ]
            # breakpoint()
            # raise NotImplementedError("Substitution def not implemented for ", children)


class SubstitutionRef(UnserializableNode):
    """
    This will be in the for \|XXX\|, and need to be replaced.
    """

    value: str

    def __init__(self, value):
        self.value = value


@register(4018)
class Unimplemented(Node):
    placeholder: str
    value: str

    def __repr__(self):
        return f"<Unimplemented {self.placeholder!r} {self.value!r}>"


from .myst_ast import (
    MText,
    MList,
    MParagraph,
    MMystDirective,
    UnprocessedDirective,
    MCode,
    MLink,
    MAdmonition,
    MMath,
    MComment,
    MBlockquote,
    MTarget,
    MThematicBreak,
    MImage,
    ReplaceNode,
)


@register(4017)
class MUnimpl(Node):
    children: List[Union[MText]]

    def __repr__(self):
        return f"<MUnimpl {self.children}>"


class IntermediateNode(Node):
    """
    This is just a dummy class for Intermediate node that should not make it to the final Product
    """

    pass


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

    def __post_init__(self):
        if self.module is not None:
            assert "." not in self.module, self.module

    def __iter__(self):
        assert isinstance(self.path, str)
        return iter([self.module, self.version, self.kind, self.path])

    @classmethod
    def from_untrusted(cls, module, version, kind, path):
        assert ":" not in module
        return cls(module, version, kind, path)


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
            # Code,
            DefList,
            FieldList,
            Fig,
            MAdmonition,
            MBlockquote,
            MCode,
            MComment,
            MList,
            MMath,
            MMystDirective,
            UnprocessedDirective,
            MParagraph,
            MTarget,
            MText,
            MThematicBreak,
            Options,
            Parameters,
            SubstitutionDef,
            SubstitutionRef,
            Unimplemented,
            MUnimpl,
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

    def extend(self, items):
        self.children.extend(items)

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
            Fig,
            DefListItem,
            DefList,
            MMystDirective,
            UnprocessedDirective,
            MMath,
            MAdmonition,
            MBlockquote,
            MList,
            MParagraph,
            MCode,
            SubstitutionDef,
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


def compress_word(stream) -> List[Any]:
    acc = []
    wds = ""
    assert isinstance(stream, list), stream
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


inline_nodes = tuple(
    [
        Directive,
        Link,
        MLink,
        SubstitutionRef,
    ]
)


@register(4021)
class TocTree(Node):
    children: List[TocTree]
    title: str
    ref: RefInfo
    open: bool = False
    current: bool = False


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
            MText,
            MCode,
        ]
    ]
    body: List[
        Union[
            MMystDirective,
            MText,
            MParagraph,
            MCode,
        ]
    ]

    def validate(self):
        for p in self.body:
            assert isinstance(p, MParagraph), p
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


@register(4033)
class DefList(Node):
    children: List[DefListItem]


@register(4037)
class DefListItem(Node):
    dt: Union[
        MParagraph,
        MText,
        MLink,
        MUnimpl,
    ]  # TODO: this is technically incorrect and should
    # be a single term, (word, directive or link is my guess).
    dd: List[
        Union[
            MParagraph,
            MCode,
            MList,
            MBlockquote,
            DefList,
            MMystDirective,
            UnprocessedDirective,
            Unimplemented,
            MUnimpl,
            MAdmonition,
            MMath,
            FieldList,
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

    # TODO: Chck why we hav a Union Here, and if we have only Paragraphs, remove the union.
    descriptions: List[Union[MParagraph]]
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


def parse_rst_section(text, qa):
    """
    This should at some point be completely replaced by tree sitter.
    in particular `from ts import parse`
    """

    from .ts import parse

    items = parse(text.encode(), qa)
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
    doc = parse_rst_section(ex, "test")
    for b in doc:
        print(b)
