import pytest

from ..take2 import (
    Block,
    BlockDirective,
    BlockVerbatim,
    DefListItem,
    Example,
    Header,
    Lines,
    Paragraph,
    dedent_but_first,
    get_object,
    make_block_3,
    parse_rst_to_papyri_tree,
)

examples = [
    (
        """
This is a block

This is a second block

""",
        3,
    ),
    (
        """
This is a block
    with a subblock

This is a second block
    with another subblock


This one:

    with a space

This one

    where the subblock

    has blank lines

and a last

""",
        6,
    ),
]


@pytest.mark.parametrize("example, nblocks", examples)
def test_make_block(example, nblocks):
    blocks = make_block_3(Lines(example.split("\n")))
    assert len(blocks) == nblocks


@pytest.mark.parametrize(
    "target, expected",
    [
        ("numpy", (0, 1, 1, 1, 1, 1)),
        pytest.param("scipy", (0, 1, 2, 2, 2), marks=[pytest.mark.xfail]),
        ("matplotlib", ()),
        ("matplotlib.pyplot.hist", (0, 0, 0, 0, 0)),
    ],
)
def test_parse_headers(target, expected):
    doc = parse_rst_to_papyri_tree(dedent_but_first(get_object(target).__doc__))
    levels = tuple([h.level for h in doc if isinstance(h, Header)])
    assert levels == expected


@pytest.mark.parametrize(
    "target, type_, number",
    [
        ("numpy", BlockDirective, 0),
        ("numpy.linspace", BlockDirective, 2),
        ("scipy.optimize._lsq.least_squares", Paragraph, 24),
        ("scipy.optimize._lsq.least_squares", Example, 14),
        ("scipy.optimize._lsq.least_squares", Header, 6),
        ("scipy.optimize._lsq.least_squares", BlockVerbatim, 1),
        ("scipy.optimize._lsq.least_squares", BlockDirective, 1),
    ],
)
def test_parse_blocks(target, type_, number):

    blocks = parse_rst_to_papyri_tree(dedent_but_first(get_object(target).__doc__))
    filtered = [b for b in blocks if type(b) == type_]
    assert len(filtered) == number
