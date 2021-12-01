from functools import lru_cache

from papyri.gen import Config, Gen, NumpyDocString


@lru_cache
def ex1():
    pass


def test_find_beyond_decorators():
    """test that we find function locations

    For example the lru_decorator.
    """
    gen = Gen(dummy_progress=True)
    doc, figs = gen.do_one_item(
        ex1,
        NumpyDocString(""),
        qa="irrelevant",
        new_config=Config(exec=True, infer=True),
    )

    assert doc.item_file.endswith("test_gen.py")
