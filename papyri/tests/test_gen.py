from pathlib import Path
from tempfile import TemporaryDirectory

from ..crosslink import Ingester
from ..gen import Gen
from ..render import _ascii_render, _route


def test_gen_numpy():
    with TemporaryDirectory() as t:
        NFUNC = 26
        t = Path(t)
        g = Gen()
        g.cache_dir = t / "cache"
        g.cache_dir.mkdir()
        g.do_one_mod(["papyri"], infer=False, exec_=False)
        import time

        num = [x.name[:-5] for x in (t / "cache").glob("papyri/*.json")]
        assert len(num) == NFUNC + 1
        assert "papyri.gen.gen_main" in num

        ing = Ingester()
        ing.cache_dir = t / "cache"
        ing.ingest_dir = t / "ingest"
        ing.ingest_dir.mkdir()
        ing.ingest("papyri", check=True)
        ing_r = [x.name[:-5] for x in (ing.ingest_dir).glob("*.json")]
        assert len(ing_r) == NFUNC, f"{set(ing_r) - set(num)} | {set(num) - set(ing_r)}"

        res = _route("papyri.gen.gen_main", ing.ingest_dir)
        assert "main entry point" in res

        assert "main entry point" in _ascii_render(
            "papyri.gen.gen_main", ing.ingest_dir
        )
