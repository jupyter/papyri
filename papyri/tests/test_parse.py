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
    pytest.raises(errors.SpaceAfterBlockDirectiveError, parse, data.encode())


def test_parse_space():
    [section] = parse(
        "Element-wise maximum of two arrays, propagating any NaNs.".encode()
    )
    assert (
        section.children[0].children[0].value
        == "Element-wise maximum of two arrays, propagating any NaNs."
    )


def test_parse_reference():
    [section] = parse("This is a `reference <to this>`_".encode())
    [paragraph] = section.children
    [text, reference] = paragraph.children
    assert reference.value == "reference <to this>"
    assert text.value == "This is a "
