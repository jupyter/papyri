from os.path import expanduser
from pathlib import Path

base_dir = Path(expanduser("~/.papyri/"))
base_dir.mkdir(parents=True, exist_ok=True)

html_dir = base_dir / "html"
html_dir.mkdir(parents=True, exist_ok=True)

ingest_dir = base_dir / "ingest"
ingest_dir.mkdir(parents=True, exist_ok=True)


logo = r"""
  ___                    _
 | _ \__ _ _ __ _  _ _ _(_)
 |  _/ _` | '_ \ || | '_| |
 |_| \__,_| .__/\_, |_| |_|
          |_|   |__/
"""
