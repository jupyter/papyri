"""
Papyri – in progress

Here are a couple of documents, or docstrings that are of interest to see how
papyri works, generally because they crashed papyri at some point during the
development, and/or do not work yet.::

You likely want to see the readme.md file for now which is kept up do date more often.


Installation
------------

pip install papyri

Usage
-----

  $ papyri gen <config file>

  $ papyri ingest ~/.papyri/data/library_version/

  $ papyri render

  $ papyri browse ....

  $ papyri serve


Of Interest
-----------

Here are a couple of function that are of interest to explore what papyri can do and render.

`dask.delayed.delayed`
    one of the parameter of the docstring has multiple paragraphs.

`IPython.core.display.Video.__init__`
    Block Verbatim in params ?

`IPython.core.interactiveshell.InteractiveShell.complete`
    contain a `DefListItem`

`matplotlib.transforms.Bbox`
    parsing of example is completely incorrect.

`matplotlib.axes._axes.Axes.text`
    misparse example as well.

`IPython.core.completer.Completion`
    item list

`matplotlib.figure.Figure.add_subplot`
    custom double dot example

`matplotlib.colors`
    unnumbered list with indent.

`matplotlib.colors`
    contain a reference via ``.. _palettable: value`` and autolink ``paletable_``.

`numpy.npv`
    hase warning sections.

`scipy.signal.ltisys.bode`
    contains multiple figure

`scipy.signal.barthann`
    multiple figures

`numpy.where`
    The Parameter section have a pair of parameter separated with the coma in
    the name field; and those parameter should be properly detected as local
    references in the rest of the docstrings.

`numpy.polynomial.laguerre.lagfit`
    should have plenty of math items


`numpy.polyfit`
    The see also section links to `scipy.interpolate.UnivariateSpline` which
    will not resolve (yet) as the fully qualified name is
    `scipy.interpolate.fitpack2.UnivariateSpline`; this should be fixed at some
    points via aliases; in the intro one link as a ``name <value>`` syntax which
    is also not yet recognized.

`scipy.signal.filter_design.zpk2sos`:
    multi blocks in enumerated list

`scipy.signal.filter_design.zpk2sos`:
    blockquote insted of enumerate list (to fix upstream)

`scipy.optimize._lsq.trf`:
    has lineblocks, which I belive is wrong.

`scipy.signal.exponential`:
    multiple figures

`numpy.einsum`:
    one of the longest numpy docstring/document, or at least one of the longest to render, with
    `scipy.signal.windows.windows.dpss` , `scipy.optimize._minimize.minimize` and
    `scipy.optimize._basinhopping.basinhopping`


Changes in behavior
-------------------

Papyri parsing might be a bit different from docutils/sphinx parsing. As
docutils try to keep backward compatibility for historical reason, we may be a
bit stricter on some of the syntax you use to. This allows us to catch more
errors. Feel free to report differences in parsing, here we document the one we
do on purpose.


Directive must not have spaces before double colon.

    .. directive :: will be seen as a comment.
            and thus this will not appear in final output.

    .. directive:: is the proper way to write block directive.
            it will be properly interpreted.


"""


import io
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import toml
import typer

from . import examples

__version__ = "0.0.8"

logo = r"""
  ___                    _
 | _ \__ _ _ __ _  _ _ _(_)
 |  _/ _` | '_ \ || | '_| |
 |_| \__,_| .__/\_, |_| |_|
          |_|   |__/
"""

app = typer.Typer(
    help="""

Generate Rich documentation for IPython, Jupyter, and publish online.

Generating Docs:

    To generate documentation IR for publishing.

    $ papyri gen examples/numpy.toml

    Will generate in ~/.papayri/data/ the folder `numpy_$numpyversion`.

Ingesting Docs:

    To crosslink a given set of IR with current existing documentation IRs

    $ papyri ingest numpy_$numpyversion [....]

Generating standalone HTML for all the known docs

    $ papyri render

To start a server that generate html on the fly

    $ papyri serve

View given function docs in text with ANSI coloring

    $ papyri ascii numpy.linspace


"""
)


def _intro():
    """
    Prints the logo and current version to stdout.

    """
    # TODO: should we print to stderr ?
    print(logo)
    print(__version__)


@app.command()
def ingest(
    paths: List[Path],
    check: bool = False,
    relink: bool = True,
    dummy_progress: bool = typer.Option(False, help="Disable rich progress bar"),
):
    """
    Given paths to a docbundle folder, ingest it into the known libraries.

    Parameters
    ----------
    paths : List of Path
        list of paths (directories) to ingest. Maybe later we want to support zipped bundled but it's the not the case
        yet.
    relink : bool
        after ingesting all the path, should we rescan the whole library to find new crosslinks ?
    """
    _intro()
    from . import crosslink as cr

    for p in paths:
        cr.main(Path(p), check, dummy_progress=dummy_progress)
    if relink:
        cr.relink()


@app.command()
def install(
    names: List[str],
    check: bool = False,
    dummy_progress: bool = typer.Option(False, help="Disable rich progress bar"),
):
    """
    WIP, download and install a remote docbundle
    """

    from io import BytesIO
    from tempfile import TemporaryDirectory

    import httpx
    import rich
    import trio
    from rich.console import Console

    from . import crosslink as cr

    console = Console()

    _intro()

    async def get(name, version, results, progress):
        """
        Utility to download a single docbundle and
        put it into result.

        """

        buf = BytesIO()

        client = httpx.AsyncClient()

        async with client.stream(
            "GET", f"https://pydocs.github.io/pkg/{name}-{version}.zip"
        ) as response:
            total = int(response.headers["Content-Length"])

            download_task = progress.add_task(f"Download {name} {version}", total=total)
            async for chunk in response.aiter_bytes():
                buf.write(chunk)
                progress.update(download_task, completed=response.num_bytes_downloaded)

            if response.status_code != 200:
                results[(name, version)] = None
            else:
                buf.seek(0)
                results[(name, version)] = buf.read()

    async def trio_main():
        """
        Main trio routine to download docbundles concurently.

        """
        results = {}
        with rich.progress.Progress(
            "{task.description}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            rich.progress.BarColumn(bar_width=None),
            rich.progress.DownloadColumn(),
            rich.progress.TransferSpeedColumn(),
        ) as progress:
            async with trio.open_nursery() as nursery:
                for name in names:
                    if "==" in name:
                        name, version = name.split("==")
                    else:
                        try:
                            mod = __import__(name)
                            version = mod.__version__
                            print(
                                f"Autodetecting version for {name}:{version}, use {name}==<version> if incorrect."
                            )
                        except Exception:
                            print(
                                f"Could not detect version for {name} use {name}==<version> if incorrect."
                            )
                            continue
                    # with console.status(f"Downloading documentation for {name}") as status:
                    nursery.start_soon(get, name, version, results, progress)
        return results

    datas = trio.run(trio_main)
    for (name, version), data in datas.items():
        if data is not None:
            # print("Downloaded", name, version, len(data) // 1024, "kb")
            zf = zipfile.ZipFile(io.BytesIO(data), "r")
            with TemporaryDirectory() as d:
                zf.extractall(d)
                cr.main(
                    next(iter([x for x in Path(d).iterdir() if x.is_dir()])),
                    check,
                    dummy_progress=dummy_progress,
                )
        else:
            print(f"Could not find docs for {name}=={version}")
    cr.relink()


@app.command()
def relink():
    """
    Rescan all the documentation to find potential new crosslinks.
    """
    _intro()
    from . import crosslink as cr

    cr.relink()


@app.command()
def gen(
    file: str,
    infer: Optional[bool] = typer.Option(
        True, help="Whether to run type inference on code examples."
    ),
    exec: Optional[bool] = typer.Option(
        None, help="Whether to attempt to execute doctring code."
    ),
    experimental: bool = typer.Option(False, help="Use experimental Ts parsing"),
    debug: bool = False,
    dummy_progress: bool = typer.Option(False, help="Disable rich progress bar"),
    dry_run: bool = False,
):
    """
    Generate documentation for a given package.

    First item should be the root package to import, if subpackages need to be
    analyzed  but are not accessible from the root pass them as extra arguments.

    """
    _intro()
    from papyri.gen import gen_main

    gen_main(
        infer=infer,
        exec_=exec,
        target_file=file,
        experimental=experimental,
        debug=debug,
        dummy_progress=dummy_progress,
        dry_run=dry_run,
    )


@app.command()
def bootstrap(file: str):
    p = Path(file)
    if p.exists():
        sys.exit(f"{p} already exists")
    name = input(f"package name [{p.stem}]:")
    if not name:
        name = p.stem
    p.write_text(toml.dumps(dict(name={"module": [name]})))


@app.command()
def render(
    ascii: bool = False, html: bool = True, dry_run: bool = False, sidebar: bool = True
):
    _intro()
    import trio

    from .render import main as m2

    trio.run(m2, ascii, html, dry_run, sidebar)


@app.command()
def ascii(name: str):
    _intro()
    import trio

    from .render import ascii_render

    trio.run(ascii_render, name)


@app.command()
def serve(sidebar: bool = True):
    _intro()
    from .render import serve as s2

    s2(sidebar=sidebar)


@app.command()
def serve_static():
    import http.server
    import socketserver

    PORT = 8000
    from papyri.config import html_dir

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(html_dir), **kwargs)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"serving at http://localhost:{PORT}")
        httpd.serve_forever()


@app.command()
def browse(qualname: str):

    from papyri.browser import main as browse

    browse(qualname)


@app.command()
def build_parser():
    from tree_sitter import Language, Parser

    pth = Path(__file__).parent / "rst.so"
    if pth.exists():
        print("parser exists, erasing to rebuild")
        pth.unlink()

    spth = str(pth)

    Language.build_library(
        # Store the library in the `build` directory
        spth,
        # Include one or more languages
        [
            "tree-sitter-rst",
        ],
    )

    PY_LANGUAGE = Language(spth, "rst")


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
    app()
