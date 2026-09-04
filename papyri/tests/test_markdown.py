"""Tests for the markdown → IR parser (:mod:`papyri.ts_markdown`)."""

from textwrap import dedent
from typing import Any

from papyri.nodes import (
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
    Text,
    ThematicBreak,
    section_title_text,
)
from papyri.ts_markdown import parse


def _parse(text: str) -> list[Section]:
    sections = parse(dedent(text).encode(), "test")
    for section in sections:
        section.validate()
    return sections


def _body(text: str) -> tuple[Any, ...]:
    """Children of the single section produced by a heading-less document."""
    [section] = _parse(text)
    assert section.title == ()
    return section.children


def test_headings_nest_by_marker_level() -> None:
    sections = _parse("""
        # Top

        text

        ## Sub

        more

        # Second
    """)
    assert [(s.level, section_title_text(s.title)) for s in sections] == [
        (0, "Top"),
        (1, "Sub"),
        (0, "Second"),
    ]


def test_setext_headings() -> None:
    sections = _parse("""
        Title
        =====

        Subtitle
        --------
    """)
    assert [(s.level, section_title_text(s.title)) for s in sections] == [
        (0, "Title"),
        (1, "Subtitle"),
    ]


def test_heading_keeps_inline_markup() -> None:
    [section] = _parse("# The `parse` function")
    assert section.title == (Text("The "), InlineCode("parse"), Text(" function"))


def test_paragraph_inline_vocabulary() -> None:
    (paragraph,) = _body("Plain *em* and **strong** and `code`.")
    assert paragraph == Paragraph(
        (
            Text("Plain "),
            Emphasis((Text("em"),)),
            Text(" and "),
            Strong((Text("strong"),)),
            Text(" and "),
            InlineCode("code"),
            Text("."),
        )
    )


def test_gap_text_between_inline_nodes_is_preserved() -> None:
    """The inline grammar emits no text nodes; prose is the byte gap."""
    (paragraph,) = _body("a *b* c *d* e")
    assert "".join(
        n.value if isinstance(n, Text) else "" for n in paragraph.children
    ) == ("a  c  e")


def test_inline_link() -> None:
    (paragraph,) = _body("see [the docs](https://example.com) here")
    assert paragraph.children[1] == Link((Text("the docs"),), "https://example.com", "")


def test_link_with_title() -> None:
    (paragraph,) = _body('[x](https://example.com "hover")')
    assert paragraph.children[0] == Link((Text("x"),), "https://example.com", "hover")


def test_reference_link_resolves_from_definition() -> None:
    (paragraph,) = _body("""
        Read the [manual] first.

        [manual]: https://example.com/manual
    """)
    assert paragraph.children[1] == Link(
        (Text("manual"),), "https://example.com/manual", ""
    )


def test_reference_link_defined_after_use() -> None:
    """Definitions are collected in a pre-pass, so order does not matter."""
    (paragraph,) = _body("""
        [late][ref]

        [ref]: https://example.com
    """)
    assert paragraph.children[0] == Link((Text("late"),), "https://example.com", "")


def test_unresolved_reference_link_degrades_to_text() -> None:
    (paragraph,) = _body("a [dangling] span")
    assert not any(isinstance(n, Link) for n in paragraph.children)
    assert "dangling" in "".join(
        n.value for n in paragraph.children if isinstance(n, Text)
    )


def test_autolink() -> None:
    (paragraph,) = _body("<https://example.com>")
    assert paragraph.children[0] == Link(
        (Text("https://example.com"),), "https://example.com", ""
    )


def test_inline_image() -> None:
    (paragraph,) = _body("![alt text](img.png)")
    assert paragraph.children[0] == Image("img.png", "alt text")


def test_badge_image_inside_link() -> None:
    """``[![badge](img)](url)`` — ubiquitous in READMEs, and the reason
    ``Image`` had to become valid phrasing content."""
    (paragraph,) = _body("[![License](badge.svg)](https://example.com/license)")
    assert paragraph.children[0] == Link(
        (Image("badge.svg", "License"),), "https://example.com/license", ""
    )


def test_emphasis_inside_link() -> None:
    (paragraph,) = _body("[**bold** link](https://example.com)")
    link = paragraph.children[0]
    assert isinstance(link, Link)
    assert link.children[0] == Strong((Text("bold"),))


def test_fenced_code_block_keeps_body() -> None:
    (code,) = _body("""
        ```python
        x = 1
        y = 2
        ```
    """)
    assert code == Code("x = 1\ny = 2")


def test_indented_code_block() -> None:
    (code,) = _body("""
        text

            indented = True
    """)[1:]
    assert isinstance(code, Code)
    assert "indented = True" in code.value


def test_code_span_folds_newlines() -> None:
    """CommonMark turns line endings inside a code span into spaces, and
    ``InlineCode`` asserts it holds a single line."""
    (paragraph,) = _body("a `wrapped\ncode` span")
    assert InlineCode("wrapped code") in paragraph.children


def test_bullet_list() -> None:
    (blist,) = _body("""
        - one
        - two
    """)
    assert blist == BulletList(
        False,
        1,
        (
            ListItem((Paragraph((Text("one"),)),)),
            ListItem((Paragraph((Text("two"),)),)),
        ),
    )


def test_ordered_list_records_start() -> None:
    (blist,) = _body("""
        3. three
        4. four
    """)
    assert blist.ordered is True
    assert blist.start == 3


def test_nested_list() -> None:
    (blist,) = _body("""
        - outer
          - inner
    """)
    (item,) = blist.children
    assert any(isinstance(child, BulletList) for child in item.children)


def test_task_list_marker_is_stripped() -> None:
    (blist,) = _body("- [x] done")
    (item,) = blist.children
    assert item == ListItem((Paragraph((Text("done"),)),))


def test_block_quote() -> None:
    (quote,) = _body("> quoted text")
    assert quote == Blockquote((Paragraph((Text("quoted text"),)),))


def test_thematic_break() -> None:
    assert _body("---\n") == (ThematicBreak(),)


def test_pipe_table() -> None:
    (table,) = _body("""
        | a | b |
        | - | - |
        | 1 | 2 |
    """)
    assert isinstance(table, Table)
    header, row = table.children
    assert header.header is True
    assert row.header is False
    assert len(header.children) == len(row.children) == 2


def test_html_block_is_dropped() -> None:
    body = _body("""
        <div class="x">raw</div>

        after
    """)
    assert body == (Paragraph((Text("after"),)),)


def test_inline_html_is_dropped() -> None:
    (paragraph,) = _body("text <br/> more")
    assert not any("<br" in n.value for n in paragraph.children if isinstance(n, Text))


def test_frontmatter_is_dropped() -> None:
    body = _body("""\
        ---
        title: Example
        ---

        body text
    """)
    assert body == (Paragraph((Text("body text"),)),)


def test_empty_document() -> None:
    assert parse(b"", "test") == []


def test_gen_visitor_accepts_markdown_ir() -> None:
    """The whole point of matching ``ts.parse``'s signature: everything
    downstream of it is format-agnostic."""
    from pathlib import Path

    from papyri.tree import GenVisitor

    sections = _parse("""
        # Title

        A paragraph with `code`, a [link](https://example.com) and
        an ![image](img.png).

        - a list
        - of items
    """)
    visitor = GenVisitor(
        "test",
        frozenset(),
        local_refs=set(),
        aliases={},
        version="0.0",
        config=None,
        module="papyri",
        doc_path=Path("."),
        asset_store=lambda name, data: None,
        doc_root=Path("."),
        doc_targets={},
        external_targets={},
        doc_titles={},
        execute=False,
    )
    for section in sections:
        visitor.visit(section).validate()
