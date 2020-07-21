import glob

import pytest

from minirst import compute_indents, find_indent_blocks, reformat

test_files = glob.glob("examples/*.rst")

expected = [f[:-3] + "expected" for f in test_files]


@pytest.mark.parametrize("test_input,expected", zip(test_files, expected))
def test_reformat_1(test_input, expected):
    with open(test_input) as f:
        inp = f.read()
    with open(expected) as f:
        exp = f.read()
    assert reformat(inp) == exp.strip("\n")


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            """this
is an
example
""",
            [0, 0, 0],
        ),
        (
            """this
  
example
""",
            [0, None, 0],
        ),
        (
            """ this
  
example
""",
            [1, None, 0],
        ),
        (
            """ this
  
  example
""",
            [1, None, 2],
        ),
    ],
)
def test_blocks(test_input, expected):
    assert compute_indents(test_input.splitlines()) == expected
