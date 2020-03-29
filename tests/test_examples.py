import pytest
from minirst import reformat

import glob
test_files = glob.glob('examples/*.rst')

expected = [f[:-3]+'expected' for f in test_files]

@pytest.mark.parametrize("test_input,expected", zip(test_files, expected))
def test_reformat_1(test_input, expected):
    with open(test_input) as f:
        inp = f.read()
    with open(expected) as f:
        exp = f.read()
    assert reformat(inp) == exp.strip('\n')
