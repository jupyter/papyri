from minirst import reformat


example1 = "This is an example text, that I'll try to reformat. It should be relatively long to make sure I can deal with long lines."
expected1 = (
"""This is an example text, that I'll try to reformat. It should be relatively long
to make sure I can deal with long lines."""
)

def test_reformat_1():
    assert reformat(example1) == expected1
