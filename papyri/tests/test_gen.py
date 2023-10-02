import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

from papyri.gen import APIObjectInfo, BlockExecutor, Config, Gen, NumpyDocString


@lru_cache
def ex1():
    pass


def test_BlockExecutor():
    b = BlockExecutor({})
    b.exec("# this is a comment")


def test_find_beyond_decorators():
    """test that we find function locations

    For example the lru_decorator.
    """
    config = Config(exec=True, infer=True)
    gen = Gen(dummy_progress=True, config=config)

    api_object = APIObjectInfo("function", "", None, None, qa=None)
    doc, figs = gen.prepare_doc_for_one_object(
        ex1,
        NumpyDocString(""),
        qa="irrelevant",
        config=config,
        aliases=[],
        api_object=api_object,
    )

    assert doc.item_file.endswith("test_gen.py")


def test_infer():
    import scipy
    from scipy._lib._uarray._backend import Dispatchable

    from papyri.gen import Config, parse_script

    c = Config(infer=True)
    res = parse_script(
        "\nx = Dispatchable(1, str)\nx",
        {"Dispatchable": Dispatchable, "scipy": scipy},
        "",
        c,
    )

    expected = (
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
    )

    assert list(res) == list(expected)


@pytest.mark.parametrize(
    "module, submodules, objects",
    [
        ("numpy", ("core",), ("numpy:array", "numpy.core._multiarray_tests:npy_sinh")),
        ("IPython", (), ("IPython:embed_kernel",)),
    ],
)
def test_numpy(module, submodules, objects):
    config = Config(exec=False, infer=False, submodules=submodules)
    gen = Gen(dummy_progress=True, config=config)

    with tempfile.TemporaryDirectory() as tempdir:
        td = Path(tempdir)
        gen.collect_package_metadata(
            module,
            relative_dir=Path("."),
            meta={},
        )
        gen.collect_api_docs(module, limit_to=objects)
        gen.partial_write(td)

        for o in objects:
            assert (td / "module" / f"{o}.json").exists()


def test_self():
    from papyri.gen import Gen, Config

    c = Config(dry_run=True, dummy_progress=True)
    g = Gen(False, config=c)
    g.collect_package_metadata("papyri", ".", {})
    g.collect_api_docs("papyri", {"papyri.examples:example1", "papyri"})
    assert g.data["papyri.examples:example1"].to_dict()["signature"] == {
        "type": "SignatureNode",
        "kind": "function",
        "parameters": [
            {
                "type": "ParameterNode",
                "name": "pos",
                "annotation": {"type": "Empty"},
                "kind": "POSITIONAL_ONLY",
                "default": {"type": "Empty"},
            },
            {
                "type": "ParameterNode",
                "name": "only",
                "annotation": {"type": "Empty"},
                "kind": "POSITIONAL_ONLY",
                "default": {"type": "Empty"},
            },
            {
                "type": "ParameterNode",
                "name": "var",
                "annotation": {"type": "Empty"},
                "kind": "POSITIONAL_OR_KEYWORD",
                "default": {"type": "Empty"},
            },
            {
                "type": "ParameterNode",
                "name": "args",
                "annotation": {"type": "Empty"},
                "kind": "POSITIONAL_OR_KEYWORD",
                "default": {"type": "Empty"},
            },
            {
                "type": "ParameterNode",
                "name": "kwargs",
                "annotation": {"type": "Empty"},
                "kind": "KEYWORD_ONLY",
                "default": {"type": "Empty"},
            },
            {
                "type": "ParameterNode",
                "name": "also",
                "annotation": {"type": "Empty"},
                "kind": "KEYWORD_ONLY",
                "default": {"data": "None", "type": "str"},
            },
        ],
        "return_annotation": {"type": "Empty"},
    }
    assert g.data["papyri"].to_dict()["signature"] is None
