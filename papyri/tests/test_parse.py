from textwrap import dedent

import pytest

from papyri import errors
from papyri.ts import parse, Node, TSVisitor, parser


# @pytest.mark.xfail(strict=True)
def test_parse_space_in_directive_section():
    data = dedent(
        """

    .. directive ::

        this type of directive is supported by docutils but
        should raise/warn in papyri.
        It may depends on the tree-sitter rst version.

    """
    )
    pytest.raises(
        errors.SpaceAfterBlockDirectiveError,
        parse,
        data.encode(),
        "test_parse_space_in_directive_section",
    )


def test_parse_directive_body():
    data1 = dedent(
        """

    .. directive:: Directive title

        This directive declares a title and content in a block separated from
        the definition by an empty new line.

    """
    )
    data2 = dedent(
        """

    .. directive:: Directive title
        This directive declares a title and content not separated by an empty
        newline.

    """
    )

    text1 = data1.strip("\n").encode()
    text2 = data2.strip("\n").encode()

    tree1 = parser.parse(text1)
    tree2 = parser.parse(text2)

    directive1 = Node(tree1.root_node).without_whitespace()
    directive2 = Node(tree2.root_node).without_whitespace()

    tsv1 = TSVisitor(text1, "test_parse_directive_body")
    tsv2 = TSVisitor(text2, "test_parse_directive_body")

    items1 = tsv1.visit(directive1)
    items2 = tsv2.visit(directive2)

    assert items1[0].name == "directive"
    assert items1[0].args == "Directive title"
    assert items1[0].options == dict()
    assert (
        items1[0].value
        == "This directive declares a title and content in a block separated from\nthe definition by an empty new line."
    )
    assert (
        " ".join([i.value for i in items1[0].children])
        == "This directive declares a title and content in a block separated from the definition by an empty new line."
    )

    assert items2[0].name == "directive"
    assert items2[0].args == "Directive title"
    assert items2[0].options == dict()
    assert (
        items2[0].value
        == "This directive declares a title and content not separated by an empty\nnewline."
    )
    assert (
        " ".join([i.value for i in items2[0].children])
        == "This directive declares a title and content not separated by an empty newline."
    )


def test_parse_warning_directive():
    data = dedent(
        """

    .. warning:: Title

        The warning directive does not admit a title.
        Just testing now.

    """
    )
    text = data.strip("\n").encode()
    tree = parser.parse(text)
    directive = Node(tree.root_node)
    tsv = TSVisitor(text, "test_parse_directive_body")
    new_node = directive.without_whitespace()
    items = tsv.visit(new_node)

    assert items[0].name == "warning"
    assert items[0].args == ""
    assert items[0].options == dict()
    assert (
        items[0].value
        == "Title The warning directive does not admit a title.\nJust testing now."
    )
    assert items[0].children == []


def test_parse_space():
    [section] = parse(
        "Element-wise maximum of two arrays, propagating any NaNs.".encode(),
        "test_parse_space",
    )
    assert (
        section.children[0].children[0].value
        == "Element-wise maximum of two arrays, propagating any NaNs."
    )


def test_parse_no_newline():
    """
    Here we test that sections of test that contain new line in the source do
    not have new line in the output. This make it simpler to render on output
    that respect newlines
    """
    data = dedent(
        """
    we want to make sure that `this
    interpreted_text` not have a newline in it and that `this
    reference`_ does not either."""
    ).encode()

    [section] = parse(data, "test_parse_space")
    text0, directive, text1, reference, text2 = section.children[0].children
    assert "\n" not in directive.value
    assert directive.value == "this interpreted_text"
    assert "\n" not in reference.value
    assert reference.value == "this reference"


def test_parse_reference():
    [section] = parse(
        "This is a `reference <to this>`_".encode(), "test_parse_reference"
    )
    [paragraph] = section.children
    [text, reference] = paragraph.children
    assert reference.value == "reference <to this>"
    assert text.value == "This is a "


def test_parse_substitution_definition():
    data1 = dedent(
        """
    A substitution definition block contains an embedded inline-compatible
    directive (without the leading ".. "), such as "image" or "replace". For
    example, the |biohazard| symbol must be used on containers used to dispose
    of medical waste.

    .. |biohazard| image :: https://upload.wikimedia.org/wikipedia/commons/c/c0/Biohazard_symbol.svg
    """
    )
    data2 = dedent(
        """
    A substitution definition block contains an embedded inline-compatible
    directive (without the leading ".. "), such as "image" or "replace". For
    example, the |biohazard| symbol must be used on containers used to dispose
    of medical waste.

    .. |biohazard| replace :: **biohazard.png**
    """
    )
    [section1] = parse(data1.encode(), "test_parse_substitution_definition")
    [_, subsdef1] = section1.children
    [node1] = subsdef1.children
    [section2] = parse(data2.encode(), "test_parse_substitution_definition")
    [_, subsdef2] = section2.children
    [node2] = subsdef2.children
    assert subsdef1.value == "|biohazard|"
    assert node1.type == "image"
    assert (
        node1.url
        == "https://upload.wikimedia.org/wikipedia/commons/c/c0/Biohazard_symbol.svg"
    )
    assert node1.alt == ""
    assert subsdef2.value == "|biohazard|"
    assert node2.type == "replace"
    assert node2.text == "**biohazard.png**"
