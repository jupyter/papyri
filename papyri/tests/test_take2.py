from ..take2 import make_block_3, Lines
import pytest


examples = [
("""
This is a block

This is a second block

""", 3),

("""
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

""", 6)
]



@pytest.mark.parametrize('example, nblocks', examples)
def test_make_block(example, nblocks):
    blocks = make_block_3(Lines(example.split("\n")))
    assert len(blocks) == nblocks


