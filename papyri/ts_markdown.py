"""Markdown → IR parser, the CommonMark counterpart of :mod:`papyri.ts`.

The entry point mirrors :func:`papyri.ts.parse` exactly — ``parse(bytes) ->
list[Section]`` — because everything downstream of that call is
format-agnostic.  :class:`papyri.tree.GenVisitor` dispatches on IR *class
names*, so markdown-produced IR flows through the same visitor as RST; the
RST-specific handlers (``replace_Directive``, ``replace_InlineRole``,
``replace_Target``, …) simply never fire.

The target is *limited markdown inclusions* — READMEs, changelogs, short
narrative pages — not a MyST implementation.  Directives have no markdown
spelling here: MyST's ``:::{note}`` syntax is out of scope (see the
"second-producer experiment" entry in ``PLAN.md``).

Two structural differences from the RST grammar shape this module:

*Two grammars.*  ``tree_sitter_markdown`` ships a block grammar
(``language()``) and a separate inline grammar (``inline_language()``).
Block parsing yields opaque ``inline`` leaves whose bytes must be re-parsed
with the second parser.

*No text nodes.*  The inline grammar emits only *named* nodes for markup;
plain text is the byte gap between them::

    inline  b'Hello *world* and `code`'
      emphasis  b'*world*'
      code_span b'`code`'

``papyri/ts.py`` solves the same problem for RST by wrapping tree-sitter
nodes and synthesising ``Whitespace`` children for the gaps.  Here the gaps
carry real content rather than whitespace, so :func:`_with_gaps` yields them
as ``Text`` directly instead of reusing that wrapper.

Raw HTML (``html_block``, ``html_tag``) is dropped with a warning: the IR is
semantic and must never carry HTML islands (see the no-raw-HTML invariant in
``PLAN.md``).  That is the single biggest fidelity loss on real-world
READMEs, and it is deliberate.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterator
from typing import Any

import tree_sitter
import tree_sitter_markdown as _tree_sitter_markdown

from .nodes import (
    Blockquote,
    BulletList,
    Code,
    Emphasis,
    Image,
    InlineCode,
    Link,
    ListItem,
    Paragraph,
    Section,
    Strong,
    Table,
    TableCell,
    TableRow,
    Text,
    ThematicBreak,
    compress_word,
)
from .ts import TreeSitterParseError, nest_sections

block_parser = tree_sitter.Parser(
    tree_sitter.Language(_tree_sitter_markdown.language())
)
inline_parser = tree_sitter.Parser(
    tree_sitter.Language(_tree_sitter_markdown.inline_language())
)

log = logging.getLogger("papyri")

# Markdown constructs papyri deliberately does not represent.  Dropped, but
# counted so ``_warn_dropped`` can report them once per document instead of
# once per occurrence (a badge-heavy README has hundreds).
_DROPPED_BLOCK = frozenset(
    {
        "html_block",
        "minus_metadata",  # YAML frontmatter
        "plus_metadata",  # TOML frontmatter
    }
)
_DROPPED_INLINE = frozenset({"html_tag"})

_REFERENCE_LINKS = frozenset(
    {"shortcut_link", "full_reference_link", "collapsed_reference_link"}
)


def _with_gaps(node: Any, buf: bytes) -> Iterator[tuple[Any | None, str]]:
    """Yield ``(child, "")`` per child, and ``(None, text)`` for byte gaps.

    The inline grammar has no text nodes — everything that is not markup is
    the span between two named children — so the gaps *are* the prose.
    """
    cursor = node.start_byte
    for child in node.children:
        if child.start_byte > cursor:
            yield None, buf[cursor : child.start_byte].decode()
        yield child, ""
        cursor = child.end_byte
    if cursor < node.end_byte:
        yield None, buf[cursor : node.end_byte].decode()


class MarkdownVisitor:
    """Walk a tree-sitter-markdown tree and emit papyri IR nodes.

    One instance per document: it holds the link-reference table and the
    tally of dropped constructs.
    """

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        # CommonMark link reference definitions (``[label]: url``) may appear
        # anywhere in the document, including *after* the links using them,
        # so they are collected in a pre-pass.
        self.link_defs: dict[str, str] = {}
        self.dropped: dict[str, int] = {}
        # Bytes of the inline fragment currently being visited; gap text is
        # sliced out of this rather than out of the whole document, because
        # the inline grammar re-parses each fragment from offset zero.
        self._inline_buf: bytes = b""

    # ------------------------------------------------------------------
    # helpers

    def _text(self, node: Any) -> str:
        return str(node.text.decode())

    def _drop(self, kind: str) -> None:
        self.dropped[kind] = self.dropped.get(kind, 0) + 1

    def _child(self, node: Any, *types: str) -> Any | None:
        return next((c for c in node.children if c.type in types), None)

    def collect_link_defs(self, node: Any) -> None:
        """Pre-pass: record every ``[label]: destination`` in the document."""
        if node.type == "link_reference_definition":
            label = self._child(node, "link_label")
            dest = self._child(node, "link_destination")
            if label is not None and dest is not None:
                key = self._text(label).strip("[]").strip().lower()
                self.link_defs.setdefault(key, self._text(dest))
        for child in node.children:
            self.collect_link_defs(child)

    # ------------------------------------------------------------------
    # inline

    def visit_inline_fragment(self, node: Any) -> list[Any]:
        """Re-parse an opaque block-grammar ``inline`` node and visit it."""
        fragment = node.text
        previous, self._inline_buf = self._inline_buf, fragment
        try:
            root = inline_parser.parse(fragment).root_node
            return compress_word(self.visit_inline(root))
        finally:
            self._inline_buf = previous

    def visit_inline(self, node: Any) -> list[Any]:
        kind = node.type

        if kind in _DROPPED_INLINE:
            self._drop(kind)
            return []

        if kind == "code_span":
            # CommonMark folds line endings inside a code span into spaces;
            # InlineCode asserts it holds a single line.
            raw = self._text(node).strip("`").strip()
            return [InlineCode(" ".join(raw.split("\n")))]

        if kind in ("emphasis", "strong_emphasis"):
            children = tuple(compress_word(self._inline_children(node)))
            return [Emphasis(children) if kind == "emphasis" else Strong(children)]

        if kind in ("inline_link", "image"):
            return self._visit_link_like(node)

        if kind in ("uri_autolink", "email_autolink"):
            url = self._text(node).strip("<>")
            return [Link((Text(url),), url, "")]

        if kind in _REFERENCE_LINKS:
            return self._visit_reference_link(node)

        if not node.children:
            return [Text(self._text(node))]

        return self._inline_children(node)

    def _inline_children(self, node: Any) -> list[Any]:
        out: list[Any] = []
        for child, gap in _with_gaps(node, self._inline_buf):
            if child is None:
                out.append(Text(gap))
            elif child.type.endswith("_delimiter"):
                # ``*``/``` ` ``` markers: structure, not content.
                continue
            else:
                out.extend(self.visit_inline(child))
        return out

    def _visit_link_like(self, node: Any) -> list[Any]:
        destination = self._child(node, "link_destination")
        url = self._text(destination) if destination is not None else ""
        label = self._child(node, "link_text", "image_description")
        children = self.visit_inline(label) if label is not None else [Text(url)]

        if node.type == "image":
            alt = "".join(c.value for c in children if isinstance(c, Text))
            return [Image(url, alt)]

        title_node = self._child(node, "link_title")
        title = self._text(title_node).strip("\"'()") if title_node is not None else ""
        return [Link(tuple(compress_word(children)), url, title)]

    def _visit_reference_link(self, node: Any) -> list[Any]:
        label = self._child(node, "link_label")
        text = self._child(node, "link_text")
        source = label if label is not None else text
        key = self._text(source if source is not None else node)
        key = key.strip("[]").strip().lower()
        children = self.visit_inline(text) if text is not None else [Text(key)]
        url = self.link_defs.get(key)
        if url is None:
            # A ``[bracketed]`` span with no matching definition is literal
            # text in CommonMark, not a broken link.
            return children
        return [Link(tuple(compress_word(children)), url, "")]

    # ------------------------------------------------------------------
    # block

    def visit_block(self, node: Any) -> list[Any]:
        kind = node.type

        if kind in ("document", "section"):
            return self._block_children(node)

        if kind == "atx_heading":
            return self._visit_atx_heading(node)

        if kind == "setext_heading":
            return self._visit_setext_heading(node)

        if kind == "paragraph":
            inline = self._child(node, "inline")
            children = self.visit_inline_fragment(inline) if inline is not None else []
            if children and children[-1] == Text(" "):
                children.pop()
            return [Paragraph(tuple(children))] if children else []

        if kind in ("fenced_code_block", "indented_code_block"):
            return [Code(self._code_body(node))]

        if kind == "block_quote":
            return [Blockquote(tuple(self._block_children(node)))]

        if kind == "list":
            return [self._visit_list(node)]

        if kind == "thematic_break":
            return [ThematicBreak()]

        if kind == "pipe_table":
            return [self._visit_table(node)]

        if kind in _DROPPED_BLOCK:
            self._drop(kind)
            return []

        if kind == "link_reference_definition":
            # Consumed by the pre-pass; the definition itself renders nothing.
            return []

        if kind == "block_continuation" or not self._text(node).strip():
            return []

        self._drop(kind)
        return []

    def _block_children(self, node: Any) -> list[Any]:
        out: list[Any] = []
        for child in node.children:
            if child.type in ("block_quote_marker", "block_continuation"):
                continue
            out.extend(self.visit_block(child))
        return out

    def _visit_atx_heading(self, node: Any) -> list[Section]:
        marker = next(c for c in node.children if c.type.startswith("atx_h"))
        # ``atx_h1_marker`` … ``atx_h6_marker``; papyri levels are 0-based.
        level = int(marker.type[len("atx_h")]) - 1
        inline = self._child(node, "inline")
        title = tuple(self.visit_inline_fragment(inline)) if inline is not None else ()
        return [Section([], self._strip_title(title), level=level)]

    def _visit_setext_heading(self, node: Any) -> list[Section]:
        underline = next(
            (c for c in node.children if c.type.startswith("setext_h")), None
        )
        level = 0 if underline is not None and "h1" in underline.type else 1
        paragraph = self._child(node, "paragraph")
        inline = self._child(paragraph, "inline") if paragraph is not None else None
        title = tuple(self.visit_inline_fragment(inline)) if inline is not None else ()
        return [Section([], self._strip_title(title), level=level)]

    def _strip_title(self, title: tuple[Any, ...]) -> tuple[Any, ...]:
        """Normalise heading inline content into valid ``Section.title``.

        Drops the trailing whitespace ``Text`` node (matching
        ``ts.visit_section``) and removes images, which ``PhrasingContent``
        excludes on purpose: a badge appended to an H1
        (``# Project ![CI](badge.svg)``) is decoration, not part of the
        heading, and a title is projected to a plain string for slugs and
        tab labels.
        """
        nodes = self._without_images(list(title))
        # Trailing whitespace is either the gap the inline grammar leaves at
        # the end of a heading, or what a stripped trailing badge left behind.
        while nodes and isinstance(nodes[-1], Text):
            trimmed = nodes[-1].value.rstrip()
            if trimmed:
                nodes[-1] = Text(trimmed)
                break
            nodes.pop()
        return tuple(nodes)

    def _without_images(self, nodes: list[Any]) -> list[Any]:
        """Recursively drop ``Image`` nodes, descending through wrappers.

        A link-wrapped badge (``[![CI](badge.svg)](ci-url)``) reaches a title
        through ``Link``, so stripping only the top level is not enough.  A
        wrapper left empty is dropped with it: a link whose entire content
        was a badge is decoration, and keeping it would put a bare URL in the
        heading text.
        """
        out: list[Any] = []
        for node in nodes:
            if isinstance(node, Image):
                self._drop("image_in_heading")
                continue
            if isinstance(node, Link):
                children = self._without_images(list(node.children))
                if children:
                    out.append(Link(tuple(children), node.url, node.title))
            elif isinstance(node, Emphasis | Strong):
                children = self._without_images(list(node.children))
                if children:
                    out.append(type(node)(tuple(children)))
            else:
                out.append(node)
        return out

    def _code_body(self, node: Any) -> str:
        content = [
            self._text(c) for c in node.children if c.type == "code_fence_content"
        ]
        body = "".join(content) if content else self._text(node)
        return body.rstrip("\n")

    def _visit_list(self, node: Any) -> BulletList:
        items = [c for c in node.children if c.type == "list_item"]
        ordered = any(
            marker.type.startswith(("list_marker_dot", "list_marker_paren"))
            for item in items
            for marker in item.children
        )
        start = 1
        if ordered and items:
            first = next(
                (m for m in items[0].children if m.type.startswith("list_marker")),
                None,
            )
            if first is not None:
                digits = self._text(first).strip().rstrip(".)")
                if digits.isdigit():
                    start = int(digits)

        children: list[ListItem] = []
        for item in items:
            body: list[Any] = []
            for child in item.children:
                if child.type.startswith("list_marker") or child.type in (
                    "block_continuation",
                    "task_list_marker_checked",
                    "task_list_marker_unchecked",
                ):
                    continue
                body.extend(self.visit_block(child))
            children.append(ListItem(tuple(body)))
        return BulletList(ordered, start, tuple(children))

    def _visit_table(self, node: Any) -> Table:
        rows: list[TableRow] = []
        for child in node.children:
            if child.type not in ("pipe_table_header", "pipe_table_row"):
                continue
            cells = tuple(
                TableCell(tuple(self.visit_inline_fragment(cell)))
                for cell in child.children
                if cell.type == "pipe_table_cell"
            )
            rows.append(TableRow(child.type == "pipe_table_header", cells))
        return Table(tuple(rows))


@functools.lru_cache(maxsize=512)
def _parse_cached(text: bytes) -> tuple[list[Section], tuple[tuple[str, int], ...]]:
    """Parse markdown into nested sections plus a tally of dropped constructs.

    Cached on the text alone, like :func:`papyri.ts._parse_cached`: ``qa`` is
    error-reporting metadata and does not affect the result, so the warning it
    labels is emitted by the caller rather than from inside the cache.
    """
    # tree-sitter-markdown's block grammar needs a terminating line ending:
    # without one the *whole document* parses as a single ERROR node and
    # yields no sections at all.  CommonMark treats end-of-input as ending
    # the final line, so normalising here matches the spec rather than
    # papering over it.
    if text and not text.endswith(b"\n"):
        text = text + b"\n"
    tree = block_parser.parse(text)
    visitor = MarkdownVisitor(text)
    visitor.collect_link_defs(tree.root_node)
    try:
        items = visitor.visit_block(tree.root_node)
    except Exception as e:
        raise TreeSitterParseError(str(e)) from e
    return nest_sections(items), tuple(sorted(visitor.dropped.items()))


def parse(text: bytes, qa: str | None = None) -> list[Section]:
    """Parse markdown ``text`` into a list of nested :class:`Section` nodes."""
    sections, dropped = _parse_cached(text)
    if dropped:
        summary = ", ".join(f"{kind}x{count}" for kind, count in dropped)
        log.warning(
            "Dropped unsupported markdown constructs in %s: %s",
            qa or "<markdown>",
            summary,
        )
    return sections
