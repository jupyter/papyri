"""
Papyri – in progress
"""
__version__ = "0.0.2"

logo = r"""
  ___                    _ 
 | _ \__ _ _ __ _  _ _ _(_)
 |  _/ _` | '_ \ || | '_| |
 |_| \__,_| .__/\_, |_| |_|
          |_|   |__/       
"""


import typer
from typing import List

app = typer.Typer()


def _intro():
    print(logo)
    print(__version__)


def main():
    app()


from pathlib import Path


@app.command()
def ingest(paths: List[Path], check: bool = False):
    _intro()
    from . import crosslink as cr

    for p in paths:
        cr.main(Path(p), check)


@app.command()
def gen(names: List[str], infer: bool, exec: bool):
    _intro()
    from papyri.gen import gen_main

    gen_main(names, infer=infer, exec_=exec)


@app.command()
def render():
    _intro()
    import trio

    from .render import main as m2

    trio.run(m2)


@app.command()
def ascii(name: str):
    _intro()
    import trio

    from .render import ascii_render

    trio.run(ascii_render, name)


@app.command()
def serve():
    _intro()
    from .render import serve

    serve()


@app.command()
def open(qualname: str):
    _intro()
    import webbrowser

    from .config import html_dir

    path = html_dir / (qualname + ".html")
    if not path.exists():
        import sys

        sys.exit("No doc for " + qualname + f" ({path})")
    print("opening", str(path))
    webbrowser.get().open("file://" + str(path), new=1)

if __name__ == "__main__":
    main()
