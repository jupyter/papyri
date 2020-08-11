from ..gen import Gen
from tempfile import TemporaryDirectory
from pathlib import Path


def test_gen_numpy():
    with TemporaryDirectory() as t:
        t = Path(t)
        g = Gen()
        g.cache_dir = t
        g.do_one_mod(['papyri'], infer=False)
        import time
        num = list(t.glob('papyri/*.json'))
        assert  len(num) == 11


