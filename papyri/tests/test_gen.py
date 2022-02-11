from functools import lru_cache

from papyri.gen import Config, Gen, NumpyDocString


@lru_cache
def ex1():
    pass


def test_find_beyond_decorators():
    """test that we find function locations

    For example the lru_decorator.
    """
    config = Config(exec=True, infer=True)
    gen = Gen(dummy_progress=True, config=config)
    doc, figs = gen.do_one_item(
        ex1,
        NumpyDocString(""),
        qa="irrelevant",
        config=config,
        aliases=[],
    )

    assert doc.item_file.endswith("test_gen.py")


def test_infer():
    import scipy
    from scipy._lib._uarray._backend import Dispatchable
    from papyri.gen import parse_script, Config

    c = Config(infer=True)
    res = parse_script(
        "\nx = Dispatchable(1, str)\nx",
        {"Dispatchable": Dispatchable, "scipy": scipy},
        "",
        c,
    )

    expected = [
        ("\n", ""),
        ("x", "scipy._lib._uarray._backend.Dispatchable"),
        (" ", ""),
        ("=", ""),
        (" ", ""),
        ("Dispatchable", "scipy._lib._uarray._backend.Dispatchable"),
        ("(", ""),
        ("1", ""),
        (",", ""),
        (" ", ""),
        ("str", "builtins.str"),
        (")", ""),
        ("\n", ""),
        ("x", "scipy._lib._uarray._backend.Dispatchable"),
    ]

    assert res == expected
