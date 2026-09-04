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
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, TypeAlias

import cbor2

from . import signature
from .node_base import REV_TAG_MAP, Node, UnserializableNode, debug, register
from .serde import get_type_hints
from .utils import dedent_but_first

register(4444)(tuple)


@debug(4003)
class InlineRole(Node):
    """Unresolved RST interpreted-text role, e.g. ``:func:`numpy.linspace```.

    ``domain`` and ``role`` mirror Sphinx domain notation (``"py"``,
    ``"func"``).  ``inventory`` is set for intersphinx
    ``:external+<inv>:…:`` references; ``None`` for same-project roles.
    Gen replaces most ``InlineRole`` nodes with ``CrossRef``; remaining
    instances are schema-in-flux (``@debug``).
    """

    value: str
    domain: str | None
    role: str | None
    # Sphinx intersphinx `:external+<inv>:…:` prefix. When set, names the
    # inventory the (domain, role) lookup targets in another project; None for
    # ordinary same-project roles.
    inventory: str | None

    def __init__(
        self,
        value: str,
        domain: str | None,
        role: str | None,
        inventory: str | None = None,
    ) -> None:
        assert "\n" not in value, f"InlineRole should not contain newline {value}"
        super().__init__(value, domain, role, inventory)

    def __hash__(self) -> int:
        return hash((tuple(self.value), self.domain, self.role, self.inventory))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InlineRole):
            return False
        return (
            (self.role == other.role)
            and (other.domain == self.domain)
            and (self.inventory == other.inventory)
            and (self.value == other.value)
        )

    def __len__(self) -> int:
        return len(self.value) + len(self.prefix) + 2

    @property
    def prefix(self) -> str:
        prefix = ""
        if self.inventory:
            prefix += ":external+" + self.inventory
        if self.domain:
            prefix += ":" + self.domain
        if self.role:
            prefix += ":" + self.role + ":"
        return prefix

    def __repr__(self) -> str:
        return f"<InlineRole {self.prefix}`{self.value}` `{self.to_dict()}`>"

    def __str__(self) -> str:
        raise NotImplementedError


@register(4002)
class CrossRef(Node):
    """
    A cross-reference produced by the gen step and resolved by ingest.

    `reference.kind` carries the resolution state:

      - "to-resolve" — placeholder emitted by gen when a best-effort
        resolution wasn't possible; ingest's relink pass is expected to
        replace the reference.
      - "missing" — ingest attempted resolution and couldn't find a
        target; render-time should present as plain text.
      - anything else ("module", "local", "api", ...) — resolved.

    `exists` is a derived property over `reference.kind`; don't store it.
    """

    value: str
    reference: RefInfo | LocalRef
    # `kind` is a classification hint carried alongside the reference (e.g. the
    # directive role at the call site). It's not a redundant copy of
    # `reference.kind`; see tree.py's toctree handler for an example where the
    # two diverge.
    kind: str

    @property
    def exists(self) -> bool:
        if self.reference is None:
            return False
        if isinstance(self.reference, LocalRef):
            return True
        return self.reference.kind not in ("to-resolve", "missing")

    def __repr__(self) -> str:
        return f"<CrossRef: {self.value=} {self.reference=} {self.kind=}>"

    def __hash__(self) -> int:
        return hash((self.value, self.reference, self.kind))


def iter_crossrefs(node: Any) -> Iterator[CrossRef]:
    """Yield every ``CrossRef`` in an IR subtree, depth-first.

    Walks ``Node`` fields via their type hints (mirroring the traversal in
    ``node_base._invalidate``) and recurses through list/tuple/dict containers;
    scalars are ignored. The IR is acyclic, so no cycle guard is needed.

    ``CrossRef.reference`` is a leaf (``LocalRef``/``RefInfo``) and never
    contains a nested ``CrossRef``, so traversal stops at each one.
    """
    if isinstance(node, CrossRef):
        yield node
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from iter_crossrefs(item)
    elif isinstance(node, dict):
        for item in node.values():
            yield from iter_crossrefs(item)
    elif isinstance(node, Node):
        for attr in get_type_hints(type(node)):  # type: ignore[arg-type]
            yield from iter_crossrefs(getattr(node, attr, None))


class Leaf(Node):
    value: str


@debug(4027)
class SubstitutionDef(Node):
    value: str
    children: tuple[Directive | UnprocessedDirective, ...]

    def __init__(
        self,
        value: str,
        children: list[Directive | UnprocessedDirective]
        | tuple[Directive | UnprocessedDirective, ...],
    ) -> None:
        self.value = value
        assert isinstance(children, (list, tuple))
        self.children = tuple(children)


@debug(4041)
class SubstitutionRef(Leaf):
    """
    This will be in the for |XXX|, and need to be replaced.
    """

    value: str


@register(4063)
class CitationReference(Node):
    """
    Inline reference to a citation, from RST source like ``[CIT2002]_``.

    ``label`` carries the citation name (e.g. ``"CIT2002"``). The renderer
    displays it as ``[label]`` and anchors it to ``#cite-<label>``.
    """

    type = "citationReference"
    label: str


@register(4064)
class Citation(Node):
    """
    Block-level citation definition, from RST source like
    ``.. [CIT2002] Book title, Author, Year.``

    ``label`` matches the identifier used in ``CitationReference`` nodes.
    ``children`` holds the body paragraphs of the citation.  The renderer
    emits this with ``id="cite-<label>"`` so ``CitationReference`` anchors
    scroll to it.
    """

    type = "citation"
    label: str
    children: tuple[Paragraph, ...]


@register(4066)
class FootnoteReference(Node):
    """
    Inline reference to a footnote, from RST source like ``[1]_``, ``[#]_``,
    ``[#name]_`` or ``[*]_``.

    ``label`` carries the raw footnote label (``"1"``, ``"#"``, ``"#name"``,
    ``"*"``). The renderer displays it as ``[label]`` and anchors it to
    ``#footnote-<label>``.
    """

    type = "footnoteReference"
    label: str


@register(4067)
class Footnote(Node):
    """
    Block-level footnote definition, from RST source like
    ``.. [1] body text`` or ``.. [#name] body text``.

    ``label`` matches the identifier used in ``FootnoteReference`` nodes
    (``"1"``, ``"#"``, ``"#name"``, ``"*"``). ``children`` holds the body
    paragraphs.  The renderer emits this with ``id="footnote-<label>"`` so
    ``FootnoteReference`` anchors scroll to it.
    """

    type = "footnote"
    label: str
    children: list[Paragraph]


@debug(4018)
class Unimplemented(Node):
    """Block-level RST construct that gen does not yet handle.

    ``placeholder`` names the directive or construct (e.g. ``"rubric"``);
    ``value`` is the raw source text.  These nodes are ``@debug`` — their
    schema may change as handlers are added.  The renderer should display
    them as a visible warning so contributors can track coverage gaps.
    """

    placeholder: str
    value: str

    def __repr__(self) -> str:
        return f"<Unimplemented {self.placeholder!r} {self.value!r}>"


@register(4072)
class DocstringSentinel(Node):
    """Sentinel placed in a doc tree when the docstring could not be parsed.

    Emitted when numpydoc fails to parse a module docstring so the viewer
    renders a visible notice instead of a blank page.
    """

    message: str


@register(4065)
class Table(Node):
    """Structured table.

    Holds an ordered sequence of ``TableRow`` children.  Header rows carry
    ``header=True``; everything else is a body row.  Produced from
    ``.. list-table::`` (see ``papyri.directives.list_table_handler``);
    grid- and simple-table RST forms still flow through as ``Code`` nodes
    until a parser for them lands.
    """

    type = "table"
    children: tuple[TableRow, ...] = field(default_factory=tuple)


@register(4068)
class TableRow(Node):
    """One row of a ``Table``.  ``header`` marks header (``<th>``) rows."""

    type = "tableRow"
    header: bool = False
    children: tuple[TableCell, ...] = field(default_factory=tuple)


@register(4069)
class TableCell(Node):
    """One cell of a ``TableRow``.  Children are flow / phrasing content."""

    type = "tableCell"
    children: tuple[FlowContent | PhrasingContent, ...] = field(default_factory=tuple)


@register(4071)
class ParamRef(Node):
    """Inline reference to a named parameter in the enclosing function's signature.

    Emitted when ``:param:`name``` is used in a docstring to call out a
    sibling parameter.  ``name`` is the bare identifier (no backticks, no
    leading stars).  The viewer highlights every ``ParamRef`` sharing the
    same ``name`` and the corresponding entry in the rendered signature.
    """

    name: str


# ---- Document AST nodes -----------------------------------------------------
#
# Previously split into two modules; merged here to eliminate a circular
# import (PLAN.md Phase 2). The node vocabulary is loosely mdast-inspired
# for the inline/flow layer and RST/Sphinx-inspired for directives,
# admonitions, and field lists. The IR is papyri-specific: no external
# conformance target.


@register(4046)
class Text(Node):
    """Literal text run with no formatting."""

    type = "text"
    value: str

    def __init__(self, value: str) -> None:
        assert isinstance(value, str)
        self.value = value
        super().__init__()


@register(4047)
class Emphasis(Node):
    """Inline emphasis (RST ``*text*``, rendered as ``<em>``)."""

    type = "emphasis"
    children: tuple[PhrasingContent, ...]


@register(4048)
class Strong(Node):
    """Inline strong emphasis (RST ``**text**``, rendered as ``<strong>``)."""

    type = "strong"
    children: tuple[PhrasingContent, ...]


@register(4049)
class Link(Node):
    """Inline hyperlink.  ``url`` is the destination; ``title`` is the
    hover text (empty string when absent).

    Children are the full inline vocabulary rather than only the static
    subset: markdown link text routinely nests emphasis and images
    (``[**bold** link](url)``, ``[![badge](img)](url)``), which RST's
    hyperlink syntax cannot express.
    """

    type = "link"
    children: tuple[LinkContent, ...]
    url: str
    title: str


@register(4050)
class Code(Node):
    """Block-level fenced code, optionally executed.

    ``execution_status`` is ``None`` when the block was not run, or a
    string status (e.g. ``"ok"``, ``"error"``) when the gen step executed
    the example.  ``out`` holds captured stdout/stderr from execution.
    """

    type = "code"
    value: str
    execution_status: str | None
    out: str

    def __init__(
        self, value: str, execution_status: str | None = None, out: str = ""
    ) -> None:
        super().__init__(value, execution_status, out)


@register(4051)
class InlineCode(Node):
    """Inline code span (RST double backticks ``code``).  Single-line only."""

    type = "inlineCode"
    value: str

    def __init__(self, value: str) -> None:
        super().__init__(value)
        assert "\n" not in value


@register(4045)
class Paragraph(Node):
    """Block-level paragraph containing inline content."""

    type = "paragraph"
    children: tuple[ParagraphContent | UnimplementedInline, ...]


@register(4053)
class BulletList(Node):
    """Ordered or unordered list.  ``ordered`` distinguishes the two;
    ``start`` is the first item number for ordered lists (typically 1)."""

    type = "list"
    ordered: bool
    start: int
    children: tuple[ListContent, ...]


@register(4054)
class ListItem(Node):
    """Single item in a ``BulletList``."""

    type = "listItem"
    children: tuple[FlowContent | PhrasingContent | DefList | UnprocessedDirective, ...]


class Directive(UnserializableNode):
    """Transient carrier for an RST directive papyri has no handler for.

    ``DirectiveVisiter`` emits one of these when it meets a directive name with
    no registered handler, purely so the *name* survives long enough to be
    reported. A ``Directive`` is deliberately **not serializable**: encoding one
    (to CBOR or JSON) raises and names the offending directive. That forces the
    maintainer to register at least a drop or verbatim handler for it (e.g.
    ``papyri.directives:drop`` to discard it, or ``papyri.directives:code_handler``
    to keep its body verbatim) before the bundle can be produced — an unhandled
    directive can never silently leak into the IR.

    ``name`` is the directive name (e.g. ``"plot"``); ``args`` is the argument
    line; ``options`` holds the ``:key: value`` option block; ``value`` is the
    raw body text; ``children`` holds any parsed child nodes.
    """

    type = "directive"
    # Unlike the in-memory intermediates (UnprocessedDirective &c.), a Directive
    # is a *terminal* node — nothing replaces it later — so its presence in a
    # tree is always a bug. Make ``validate()`` reject it (see node_base) so gen
    # fails fast instead of only at write/pack time.
    _reject_at_validate: ClassVar[bool] = True
    name: str
    args: str | None
    options: dict[str, str]
    value: str | None
    children: tuple[FlowContent | PhrasingContent | None, ...] = field(
        default_factory=tuple
    )

    def _why_unserializable(self) -> str:
        return (
            f"refusing to serialize unhandled directive {self.name!r}: papyri has "
            f"no handler registered for it, so its content has no representation "
            f"in the IR. Register a handler for {self.name!r} in the "
            f"[global.directives] table of your papyri config — e.g. "
            f"{self.name!r} = 'papyri.directives:drop' to drop it, or "
            f"'papyri.directives:code_handler' to keep the body verbatim."
        )

    @classmethod
    def from_unprocessed(cls, up: UnprocessedDirective) -> Directive:
        return cls(up.name, up.args, up.options, up.value, up.children)


class UnprocessedDirective(UnserializableNode):
    """
    Placeholder for yet unprocessed directives, after they are parsed by
    tree-sitter but before they are dispatched through the role resolution.
    """

    name: str
    args: str | None
    options: dict[str, str]
    value: str | None
    children: tuple[FlowContent | PhrasingContent | None, ...]
    raw: str


@register(4055)
class AdmonitionTitle(Node):
    """Title of an ``Admonition`` block (e.g. the ``"Note"`` heading)."""

    type = "admonitionTitle"
    children: tuple[PhrasingContent | None, ...] = field(default_factory=tuple)


# Finite set of styling categories an admonition maps to. The viewer styles
# by `base_type` (a small, fixed vocabulary) rather than the open-ended `kind`,
# so the renderer never has to know every kind string. Gen owns this
# classification (see PLAN.md "gen owns all ref classification").
ADMONITION_BASE_TYPES: frozenset[str] = frozenset(
    {"note", "tip", "important", "warning", "danger", "neutral"}
)

# Map each admonition `kind` papyri emits to its base styling category. The
# version-change kinds map to "neutral" — they read as informational notices
# rather than a distinct severity. Unknown kinds fall back to "note".
_ADMONITION_KIND_TO_BASE_TYPE: dict[str, str] = {
    "note": "note",
    "seealso": "note",
    "topic": "note",
    "admonition": "note",
    "rubric": "note",
    "tip": "tip",
    "hint": "tip",
    "important": "important",
    "warning": "warning",
    "attention": "warning",
    "caution": "warning",
    "danger": "danger",
    "error": "danger",
    "versionadded": "neutral",
    "versionchanged": "neutral",
    "deprecated": "neutral",
}


def admonition_base_type(kind: str) -> str:
    """Return the finite base styling category for an admonition ``kind``.

    The result is guaranteed to be a member of ``ADMONITION_BASE_TYPES``;
    a mapping-table entry pointing at an unknown category raises ``ValueError``
    so a typo fails fast at gen time rather than producing an unstyleable node.
    """
    base_type = _ADMONITION_KIND_TO_BASE_TYPE.get(kind, "note")
    if base_type not in ADMONITION_BASE_TYPES:
        raise ValueError(
            f"admonition kind {kind!r} maps to unknown base_type {base_type!r}; "
            f"expected one of {sorted(ADMONITION_BASE_TYPES)}"
        )
    return base_type


@register(4056)
class Admonition(Node):
    """Block-level admonition (note, warning, tip, …).

    ``kind`` is the admonition type string (``"note"``, ``"warning"``,
    etc.).  The first child is typically an ``AdmonitionTitle``.

    ``base_type`` is the finite styling category (one of
    ``ADMONITION_BASE_TYPES``) derived from ``kind``; it is stored in the IR
    so the renderer styles by a small fixed vocabulary without replicating
    the kind→category map.
    """

    type = "admonition"
    kind: str = "note"
    base_type: str = "note"
    children: tuple[FlowContent | AdmonitionTitle | Unimplemented | DefList, ...] = (
        field(default_factory=tuple)
    )

    def _post_deserialise(self) -> None:
        self.base_type = admonition_base_type(self.kind)


class Comment(Node):
    """RST comment node.

    Kept in the Python IR and JSON serialization so post-processing tools
    can see them. Stripped during CBOR pack (see ``Node.cbor``) so they
    never appear in published bundles — hence no ``@register`` tag.
    """

    _drop_in_cbor: ClassVar[bool] = True

    type = "comment"
    value: str


@register(4058)
class Math(Node):
    """Block-level LaTeX math expression.  ``value`` is raw LaTeX."""

    type = "math"
    value: str


@register(4057)
class InlineMath(Node):
    """Inline LaTeX math expression.  ``value`` is raw LaTeX."""

    type = "inlineMath"
    value: str


@register(4059)
class Blockquote(Node):
    """Indented block quote."""

    type = "blockquote"
    children: tuple[FlowContent, ...] = field(default_factory=tuple)


@register(4061)
class Target(Node):
    """RST hyperlink target or internal anchor.

    ``label`` is the anchor name (used by ``CrossRef`` for same-page
    jumps).  ``url`` is set for external-URL targets; ``None`` for
    internal anchors.
    """

    type = "target"
    label: str
    url: str | None

    def __init__(self, label: str, url: str | None = None) -> None:
        super().__init__(label, url)


@register(4062)
class Image(Node):
    """Image.  ``url`` is the asset path; ``alt`` is the alt text.

    Block-level from RST (``.. image::`` is a directive) and inline from
    markdown (``![alt](url)``).  It is therefore in ``FlowContent``,
    ``ParagraphContent`` and ``LinkContent`` — but *not* in
    ``PhrasingContent``, which would also make it legal in a
    ``Section.title``.
    """

    type = "image"
    url: str
    alt: str


@register(4019)
class ThematicBreak(Node):
    """Horizontal rule (RST ``----`` transition)."""

    type = "thematicBreak"


@register(4001)
class Root(Node):
    """Top-level document node; the root of every IR document tree."""

    type = "root"
    children: tuple[
        FlowContent
        | Parameters
        | Unimplemented
        | SubstitutionDef
        | signature.SignatureNode
        | Image,
        ...,
    ]


@debug(4017)
class UnimplementedInline(Node):
    """Inline RST construct that gen does not yet handle.

    Like ``Unimplemented`` but for inline (phrasing) content.  Schema is
    ``@debug``; these nodes should shrink as more roles are implemented.
    """

    children: tuple[Text, ...]

    def __repr__(self) -> str:
        return f"<UnimplementedInline {self.children}>"


@register(4022)
@dataclass(frozen=True)
class LocalRef(Node):
    """
    A reference to a document within the same bundle (same package + version).

    Unlike ``RefInfo``, ``LocalRef`` omits module and version because they are
    always inherited from the bundle context.  The renderer can construct the
    full URL by combining the bundle's (module, version) with (kind, path).

    The link is guaranteed to exist: gen validates that the target is present
    before writing it.

    Parameters
    ----------
    kind : str
        Document kind: ``"docs"``, ``"module"``, ``"examples"``, etc.
    path : str
        Path within the bundle (e.g. ``"numpy.linspace"`` or ``"tutorial:index"``).
    """

    kind: str
    path: str

    def __iter__(self) -> Iterator[str]:
        return iter([self.kind, self.path])


@register(4024)
class Figure(Node):
    """Image figure whose asset is identified by a ``RefInfo`` cross-reference.

    The renderer resolves ``value`` against the bundle's asset store to
    obtain the actual image URL.
    """

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

    module: str | None
    version: str | None
    kind: str
    path: str

    def __post_init__(self) -> None:
        if self.module is not None:
            assert "." not in self.module, self.module

    def __iter__(self) -> Iterator[str | None]:
        assert isinstance(self.path, str)
        return iter([self.module, self.version, self.kind, self.path])

    @classmethod
    def from_untrusted(cls, module: str, version: str, kind: str, path: str) -> RefInfo:
        assert ":" not in module
        return cls(module, version, kind, path)


@register(4012)
class NumpydocExample(Node):
    """Numpydoc ``Examples`` section; ``value`` holds the raw example lines."""

    value: tuple[str, ...]
    title = "Examples"


@register(4013)
class NumpydocSeeAlso(Node):
    """Numpydoc ``See Also`` section; ``value`` is a sequence of ``SeeAlsoItem``
    nodes."""

    value: tuple[SeeAlsoItem, ...]
    title = "See Also"


@register(4014)
class NumpydocSignature(Node):
    """Numpydoc ``Signature`` section; ``value`` is the raw signature string."""

    value: str
    title = "Signature"


@register(4015)
class Section(Node):
    """Document section (heading + body).

    ``title`` is a tuple of inline nodes; an empty tuple means the section
    has no explicit heading (anonymous wrapper sections emitted by numpydoc
    parsers).  ``level`` is the nesting depth (0 = top-level).  ``target``
    is the RST anchor label, if one was defined above this heading.
    """

    children: tuple[
        SectionContent,
        ...,
    ]
    # Inline content (Text, InlineCode, InlineRole, Link, ...).  Empty tuple
    # means the section has no title (anonymous wrapper section emitted by
    # numpydoc-style parsers); a single Text node is the equivalent of the old
    # plain-string title.
    title: tuple[PhrasingContent, ...] = field(default_factory=tuple)
    level: int = 0
    target: str | None = None

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other)

    def __getitem__(self, k: int) -> Any:
        return self.children[k]

    def __setitem__(self, k: int, v: Any) -> None:
        lst = list(self.children)
        lst[k] = v
        self.children = tuple(lst)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.children)

    def append(self, item: Any) -> None:
        self.children = (*self.children, item)

    def extend(self, items: Any) -> None:
        self.children = (*self.children, *items)

    def empty(self) -> bool:
        return len(self.children) == 0

    def __bool__(self) -> bool:
        return len(self.children) >= 0

    def __len__(self) -> int:
        return len(self.children)


def section_title_text(title: tuple[PhrasingContent, ...]) -> str:
    """Plain-text projection of a Section.title for places that need a string
    (tab title, slug source, comparison against canonical numpydoc section
    names). Walks the inline tree and concatenates text content."""
    out: list[str] = []
    for n in title:
        v = getattr(n, "value", None)
        if isinstance(v, str):
            out.append(v)
        else:
            kids = getattr(n, "children", None)
            if kids:
                out.append(section_title_text(tuple(kids)))
    return "".join(out)


@register(4026)
class Parameters(Node):
    """Container for a numpydoc parameter list (Parameters, Returns, etc.).
    Must have at least one ``DocParam`` child."""

    children: tuple[DocParam, ...]

    def validate(self) -> Self:
        assert len(self.children) > 0
        return super().validate()


@register(4016)
class DocParam(Node):
    """Single parameter/return entry inside a ``Parameters`` block.

    ``name`` is the parameter name; ``annotation`` is its type string
    (empty string when absent); ``desc`` holds the description body.
    """

    name: str
    annotation: str
    desc: tuple[
        Figure
        | DefListItem
        | DefList
        | Directive
        | UnprocessedDirective
        | Math
        | Admonition
        | Blockquote
        | BulletList
        | Paragraph
        | Code
        | SubstitutionDef
        | Table,
        ...,
    ]

    @property
    def children(self) -> Any:
        return self.desc

    @children.setter
    def children(self, values: Any) -> None:
        self.desc = values

    def __getitem__(self, index: int) -> Any:
        return [self.name, self.annotation, self.desc][index]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name=}, {self.annotation=}, {self.desc=}>"


class GenToken(UnserializableNode):
    value: str
    qa: str | None
    pygmentclass: str


class GenCode(UnserializableNode):
    """
    Gen-time bundle of syntax-highlighted code tokens and execution output.

    Emitted while scraping docstring examples, rewritten to Code by the
    gen visitor before anything reaches disk. Present in the IR only as
    an in-memory intermediate; serialization is asserted-unreachable.
    """

    entries: tuple[GenToken, ...]
    out: str
    ce_status: str

    def validate(self) -> Self:
        for x in self.entries:
            assert isinstance(x, GenToken)

        return super().validate()

    def _validate(self) -> None:
        for _ in self.entries:
            pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.entries=} {self.out=} {self.ce_status=}>"


def compress_word(stream: list[Any]) -> list[Any]:
    acc = []
    wds = ""
    assert isinstance(stream, list), stream
    for item in stream:
        if isinstance(item, Text):
            wds += item.value
        else:
            if type(item).__name__ == "Whitespace":
                acc.append(Text(item.value))
                wds = ""
            else:
                if wds:
                    acc.append(Text(wds))
                    wds = ""
                acc.append(item)
    if wds:
        acc.append(Text(wds))
    return acc


inline_nodes = tuple(
    [
        InlineRole,
        CrossRef,
        Link,
        SubstitutionRef,
    ]
)


@register(4021)
class TocTree(Node):
    """One node in the table-of-contents tree.

    ``ref`` points to the document this entry links to.  ``children``
    holds sub-entries.
    """

    children: tuple[TocTree, ...]
    title: str
    ref: LocalRef


@debug(4034)
class Options(Node):
    """Directive option block (key-value pairs) as raw strings.  ``@debug``
    — schema may change once a structured representation is settled."""

    values: tuple[str, ...]


@register(4035)
class FieldList(Node):
    """RST field list (e.g. Sphinx ``:param x:`` blocks)."""

    children: tuple[FieldListItem, ...]


@register(4036)
class FieldListItem(Node):
    """Single entry in a ``FieldList``.  ``name`` is the field label
    (e.g. ``param x``); ``body`` holds the description."""

    name: tuple[Text | Code, ...]
    body: tuple[Directive | Text | Paragraph | Code, ...]

    def validate(self) -> Self:
        for p in self.body:
            assert isinstance(p, Paragraph), p
        if self.name:
            assert len(self.name) == 1, (self.name, [type(n) for n in self.name])
        return super().validate()

    @property
    def children(self) -> list[Any]:
        return [*self.name, *self.body]

    @children.setter
    def children(self, value: Any) -> None:
        x, *y = value
        self.name = (x,)
        self.body = tuple(y)


@register(4033)
class DefList(Node):
    """RST definition list; a sequence of term/definition pairs."""

    children: tuple[DefListItem, ...]


@register(4037)
class DefListItem(Node):
    """One term/definition pair in a ``DefList``.

    ``dt`` is the term (definition list term); ``dd`` holds the
    definition body paragraphs.
    """

    dt: (
        Paragraph | Text | Link | UnimplementedInline
    )  # TODO: this is technically incorrect and should
    # be a single term, (word, directive or link is my guess).
    dd: tuple[
        Paragraph
        | Code
        | BulletList
        | Blockquote
        | DefList
        | Directive
        | UnprocessedDirective
        | Unimplemented
        | UnimplementedInline
        | Admonition
        | Math
        | FieldList
        | TocTree
        | Table
        | Target,  # TODO: maybe remove that.
        ...,
    ]

    @property
    def children(self) -> list[Any]:
        return [self.dt, *self.dd]

    @children.setter
    def children(self, value: Any) -> None:
        self.dt, *self.dd = value
        self.validate()


@register(4028)
class SeeAlsoItem(Node):
    name: CrossRef

    descriptions: tuple[Paragraph, ...]
    # there are a few case when the lhs is `:func:something`... in scipy.
    type: str | None

    @property
    def children(self) -> list[Any]:
        return [self.name, self.type, *self.descriptions]

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.descriptions)))

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}: {self.name} {self.type} {self.descriptions}>"
        )


def get_object(qual: str) -> Any:
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


def parse_rst_section(text: str, qa: str) -> list[SectionContent]:
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
        return list(section.children)
    raise ValueError("Multiple sections present")


# ---- Union type aliases -----------------------------------------------------

StaticPhrasingContent: TypeAlias = (
    Text
    | InlineCode
    | InlineMath
    | InlineRole
    | CrossRef
    | ParamRef
    | CitationReference
    | FootnoteReference
    | SubstitutionRef
    | Unimplemented
)

PhrasingContent: TypeAlias = StaticPhrasingContent | Emphasis | Strong | Link

# Link text: everything inline except a nested Link, which markdown and RST
# both forbid.  ``Image`` is admitted because ``[![badge](img)](url)`` is in
# nearly every README.
LinkContent: TypeAlias = StaticPhrasingContent | Emphasis | Strong | Image

# Paragraph body: phrasing plus ``Image``.  Markdown's ``![alt](url)`` is
# genuinely inline and can sit mid-sentence, but ``Image`` is deliberately
# kept out of ``PhrasingContent`` itself so it cannot appear in a
# ``Section.title``.
ParagraphContent: TypeAlias = PhrasingContent | Image

FlowContent: TypeAlias = (
    Code
    | Paragraph
    | UnprocessedDirective
    | ThematicBreak
    | Blockquote
    | BulletList
    | Target
    | Directive
    | Admonition
    | Math
    | DefList
    | DefListItem
    | FieldList
    | Comment
    | Citation
    | Footnote
    | Image
    | Figure
    | Table
)

ListContent: TypeAlias = ListItem

SectionContent: TypeAlias = (
    DefList
    | FieldList
    | Figure
    | Image
    | Admonition
    | Blockquote
    | Code
    | Comment
    | BulletList
    | Math
    | Directive
    | UnprocessedDirective
    | Paragraph
    | Target
    | Text
    | ThematicBreak
    | Options
    | Parameters
    | SubstitutionDef
    | SubstitutionRef
    | Unimplemented
    | UnimplementedInline
    | DocstringSentinel
    | Citation
    | Footnote
    | Table
)


class Encoder:
    def __init__(self, rev_map: dict[int, Any]) -> None:
        self._rev_map = rev_map

    def encode(self, obj: Any) -> bytes:
        # canonical=True sorts map keys per RFC 8949 §4.2, so the same logical
        # input always produces byte-identical CBOR. Node fields are encoded as
        # CBOR arrays (see node_base.Node.cbor), so attribute order is fixed by
        # the class definition. The only dict whose iteration order is
        # semantic (GeneratedDoc._content) has its order carried separately
        # via _ordered_sections, so sorting its keys here loses no information.
        out: bytes = cbor2.dumps(
            obj,
            default=lambda encoder, obj: obj.cbor(encoder),
            canonical=True,
        )
        return out

    def _type_from_tag(self, tag: Any) -> Any:
        return self._rev_map[tag.tag]

    def _tag_hook(self, *args: Any, **_kwargs: Any) -> Any:
        # cbor2 has shifted calling conventions for tag_hook across major
        # versions: 5.x calls (decoder, tag[, shareable_index]); 6.x calls
        # (tag, immutable) without the decoder. We don't use any of the
        # extras, so just pick the CBORTag out of whichever positional slot
        # it landed in.
        #
        # cbor2 ≥ 6 decodes containers inside tagged values as immutable:
        # CBOR arrays → tuples (correct: node list fields are now tuple[T, ...])
        # CBOR maps   → frozendict (not a dict subclass) → convert to dict.
        from cbor2 import CBORTag

        tag = next(a for a in args if isinstance(a, CBORTag))
        type_ = self._type_from_tag(tag)
        tt = get_type_hints(type_)
        kwds = {}
        for (k, ann), v in zip(tt.items(), tag.value, strict=False):
            if getattr(ann, "__origin__", None) is dict and not isinstance(v, dict):
                # cbor2 ≥ 6 gives frozendict for CBOR maps; dict() accepts any mapping.
                v = dict(v)
            kwds[k] = v
        return type_(**kwds)

    def decode(self, data: bytes) -> Any:
        return cbor2.loads(data, tag_hook=self._tag_hook)

    def _available_tags(self) -> set[int]:
        k = self._rev_map.keys()
        mi, ma = min(k), max(k)
        return set(range(mi, ma + 2)) - set(k)


encoder = Encoder(REV_TAG_MAP)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "numpy"
    ex = get_object(what).__doc__
    ex = dedent_but_first(ex)
    doc = parse_rst_section(ex, "test")
    for b in doc:
        print(b)
