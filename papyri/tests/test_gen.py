from pathlib import Path
from tempfile import TemporaryDirectory

from ..crosslink import Ingester
from ..gen import Gen
from ..render import Store, _ascii_render, _route


async def test_gen_numpy():
    with TemporaryDirectory() as t:
        NFUNC = 37
        t = Path(t)
        g = Gen()
        g.do_one_mod(["papyri"], infer=False, exec_=False, conf={})
        g.write(t)

        num = [x.name[:-5] for x in (t).glob("papyri/*.json")]
        assert len(num) == NFUNC + 1
        assert "papyri.gen.gen_main" in num
        ing = Ingester()
        ing.ingest_dir = t / "ingest"
        ing.ingest_dir.mkdir()
        ing.ingest(t, check=True)
        ing_r = [x.name[:-5] for x in (ing.ingest_dir).glob("*.json")]
        assert len(ing_r) == NFUNC, f"{set(ing_r) - set(num)} | {set(num) - set(ing_r)}"

        res = await _route("papyri.gen.gen_main", Store(ing.ingest_dir))
        assert "main entry point" in res

        assert "main entry point" in await _ascii_render(
            "papyri.gen.gen_main", Store(ing.ingest_dir)
        )
