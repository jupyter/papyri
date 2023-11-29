from textwrap import dedent

import pytest

from papyri import errors
from papyri.ts import parse


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
