"""
Various ascii rendering test, where we compare the rendered version to an expected file
"""

import pytest
from pathlib import Path
from papyri.render import GraphStore, ingest_dir, _rich_ascii

HERE = Path(__file__).parent

expected = (HERE / "expected").glob("*")


def _get_result_for_name(name):
    gstore = GraphStore(ingest_dir, {})
    key = next(iter(gstore.glob((None, None, "module", name))))

    return _rich_ascii(key, gstore)


@pytest.mark.postingest
@pytest.mark.parametrize("file", expected)
def test_g(file):
    item = file.name[: -len(".expected")]
    res = _get_result_for_name(item)

    expected = file.read_text()
    assert expected == res


if __name__ == "__main__":
    gstore = GraphStore(ingest_dir, {})
    for file in expected:
        assert file.name.endswith(".expected")
        item = file.name[: -len(".expected")]
        key = next(iter(gstore.glob((None, None, "module", item))))
        res = _get_result_for_name(item)
        print("regen", key)
        file.write_text(res)
