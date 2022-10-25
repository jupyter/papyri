import logging

from papyri.config import html_dir
import pytest

log = logging.getLogger("papyri")

PRESENT_PATTERNS = [
    (
        "p/numpy/1.23.4/api/numpy.html",
        "Make sure there is no space between verbabim and surrounding words",
        "profile (<code class='verbatim'>ipython -p numpy</code>),",
    ),
]


@pytest.mark.parametrize("path, reason, expected", PRESENT_PATTERNS)
def test_render(path, reason, expected):
    html_path = (html_dir / path)
    if not html_path.exists():
        logging.info("papyri render not done, rendering now.")
        from papyri.render import main as m2
        import trio
        trio.run(m2, False, True, False, True, True, False)
    text = html_path.read_text()
    assert expected in text, reason
