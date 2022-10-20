from papyri.config import html_dir
import pytest


PRESENT_PATTERNS = [
    (
        "p/numpy/1.23.4/api/numpy.html",
        "Make sure there is no space between verbabim and surrounding words",
        "profile (<code class='verbatim'>ipython -p numpy</code>),",
    ),
]


@pytest.mark.parametrize("path, reason, expected", PRESENT_PATTERNS)
def test_render(path, reason, expected):
    text = (html_dir / path).read_text()
    assert expected in text, reason
