import pytest

from papyri.ts import parse

from ..myst_ast import UnprocessedDirective
from ..take2 import (
    dedent_but_first,
    get_object,
)


@pytest.mark.parametrize(
    "target, type_, number",
    [
        ("numpy", UnprocessedDirective, 0),
        ("numpy.linspace", UnprocessedDirective, 2),
    ],
)
def test_parse_blocks(target, type_, number):
    sections = parse(dedent_but_first(get_object(target).__doc__).encode(), "test")
    filtered = [b for section in sections for b in section.children if type(b) == type_]
    assert len(filtered) == number
