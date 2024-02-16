from pathlib import Path
import json
from papyri.tree import DVR
from papyri.ts import parse
import sys

HERE = Path(__file__).parent
CORP = HERE / "corpus"
SAMPLES = CORP.glob("*.sample.txt")


def _expected_path(sample_path: Path) -> Path:
    return CORP / sample_path.name.replace(".sample.", ".expected.")


def _serialize(data) -> bytes:
    processed = [s.to_dict() for s in data]
    return json.dumps(processed).encode()


def _process(sample: Path):
    bytes_ = sample.read_bytes()
    data = parse(bytes_, sample)
    dv = DVR(
        str(sample),
        frozenset(),
        local_refs=set(),
        substitution_defs={},
        aliases={},
        version="TestSuite",
        config={},
    )
    return [dv.visit(s) for s in data]


if __name__ == "__main__":
    targets = [Path(p) for p in sys.argv[1:]]
    for p in targets:
        if CORP not in p.resolve().parents:
            print(p, "not in corpus", CORP, p.parents)
        else:
            print("process...", p)
            data = _process(p)
            bytes_ = _serialize(data)
            Path(_expected_path(p)).write_bytes(bytes_)
