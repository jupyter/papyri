from pathlib import Path
from tempfile import TemporaryDirectory

from ..crosslink import Ingester, load_one, IngestedBlobs
from ..gen import Gen
from ..render import Store, _ascii_render, _route


async def test_gen_papyri():
    with TemporaryDirectory() as t:
        NFUNC = 35
        t = Path(t)
        g = Gen()
        g.do_one_mod(["papyri"], infer=False, exec_=False, conf={})
        g.write(t)

        num = [x.name[:-5] for x in (t).glob("papyri/*.json")]
        assert len(num) == NFUNC + 2
        assert "papyri.gen.gen_main" in num

        ing = Ingester()
        ing.ingest_dir = t / "ingest"
        ing.ingest_dir.mkdir()
        ing.ingest(t, check=True)
        ing_r = [x.name[:-5] for x in (ing.ingest_dir).glob("*/*/*.json")]
        assert (
            len(ing_r) == NFUNC + 2
        ), f"{set(ing_r) - set(num)} | {set(num) - set(ing_r)}"

        res = await _route("papyri.gen.gen_main", Store(ing.ingest_dir))
        assert "main entry point" in res

        assert "main entry point" in await _ascii_render(
            "papyri.gen.gen_main", Store(ing.ingest_dir)
        )


async def test_gen_numpy():
    with TemporaryDirectory() as t:
        t = Path(t)
        g = Gen()
        g.do_one_mod(["numpy"], infer=False, exec_=False, conf={})
        g.write(t)

        ing = Ingester()
        ing.ingest_dir = t / "ingest"
        ing.ingest_dir.mkdir()
        ing.ingest(t, check=True)

        linspace_file = ing.ingest_dir / "numpy" / "numpy" / "numpy.linspace.json"
        assert linspace_file.exists()
        linspace_bytes = linspace_file.read_text()
        import json

        data = json.loads(linspace_bytes)
        doc_blob = load_one(linspace_bytes, None)
