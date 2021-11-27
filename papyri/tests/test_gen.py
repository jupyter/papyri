from functools import lru_cache

from papyri.gen import Gen, NumpyDocString


@lru_cache
def ex1():
    pass


def test_find_beyond_decorators():
    """test that we find function locations

    For example the lru_decorator.
    """
    gen = Gen()
    doc, figs = gen.do_one_item(
        ex1, NumpyDocString(""), infer=False, exec_=False, qa="irrelevant", config={}
    )

    assert doc.item_file.endswith("test_gen.py")
