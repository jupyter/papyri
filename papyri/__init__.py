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

app = typer.Typer(
    help="""

Generate Rich documentation for IPython, Jupyter, and publish online. 

Generating Docs:

    To generate documentation IR for publishing. 

    $ papyri gen numpy

    Will generate in CWD the folder `numpy_$numpyversion`.

Ingesting Docs:

    To crosslink a given set of IR with current existing documentation IRs

    $ papyri ingest numpy_$numpyversion [....]

Generating standalone HTML for all the kown docs

    $ papyri render

To start a server that generate html on the fly
    
    $ papyri serve

View w given function docs in text with ansi coloring

    $ papyri ascii numpy.linspace


"""
)


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
def gen(
    names: List[str],
    infer: bool = typer.Option(
        True, help="Whether to run type inference on code examples."
    ),
    exec: bool = typer.Option(
        False, help="Whether to attempt to execute doctring code."
    ),
):
    """
    Generate documentation for a given package.

    First item should be the root package to import, if subpackages need to be
    analyzed  but are not accessible from the root pass them as extra arguments.

    """
    _intro()
    from papyri.gen import gen_main

    gen_main(names, infer=infer, exec_=exec)


@app.command()
def render(ascii: bool = False, html: bool = True, dry_run: bool = False):
    _intro()
    import trio

    from .render import main as m2

    trio.run(m2, ascii, html, dry_run)


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
