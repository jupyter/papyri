import pytest

from papyri.ts import parse

from ..take2 import (
    BlockDirective,
    dedent_but_first,
    get_object,
)


@pytest.mark.parametrize(
    "target, type_, number",
    [
        ("numpy", BlockDirective, 0),
        ("numpy.linspace", BlockDirective, 2),
        # ("scipy.optimize._lsq.least_squares", Paragraph, 24),
        # ("scipy.optimize._lsq.least_squares", Example, 14),
        # ("scipy.optimize._lsq.least_squares", Header, 6),
        # ("scipy.optimize._lsq.least_squares", BlockVerbatim, 1),
        # ("scipy.optimize._lsq.least_squares", BlockDirective, 1),
    ],
)
def test_parse_blocks(target, type_, number):

    sections = parse(dedent_but_first(get_object(target).__doc__).encode())
    filtered = [b for section in sections for b in section.children if type(b) == type_]
    assert len(filtered) == number
