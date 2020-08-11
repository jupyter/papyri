from ..gen import Gen
from tempfile import TemporaryDirectory
from pathlib import Path

from ..crosslink import Ingester


def test_gen_numpy():
    with TemporaryDirectory() as t:
        t = Path(t)
        g = Gen()
        g.cache_dir = t / 'cache'
        g.cache_dir.mkdir()
        g.do_one_mod(['papyri'], infer=False)
        import time
        num = [x.name[:-5] for x in (t/'cache').glob('papyri/*.json')]
        assert  len(num) == 20
        assert 'papyri.gen.gen_main' in num

        ing = Ingester()
        ing.cache_dir = t / 'cache'
        ing.ingest_dir = t / 'ingest'
        ing.ingest_dir.mkdir()
        ing.ingest('papyri', check=True)


