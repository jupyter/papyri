"""
Papyri – in progress

Here are a couple of documents, or docstrings that are of interest to see how
papyri works, generally because they crashed papyri at some point during the
development, and/or do not work yet.::

You likely want to see the readme.md file for now which is kept up do date more often.


Installation
------------

.. code::

    pip install papyri


dev install
-----------

You may need to get a modified version of numpydoc depending on the stage of development.


    git clone https://github.com/jupyter/papyri
    cd papyri
    pip install -e .

Build the TreeSitter rst parser:

Some functionality require ``tree_sitter_rst``, see run ``papyri build-parser``, and CI config file on how to build the tree-sitter
shared object locally::

    git clone https://github.com/stsewd/tree-sitter-rst
    papyri build-parser


Usage
-----

There are mostly 3 stages to run papyri, so far you need to do the 3 steps/roles
on your local machine, but it will not be necessary once papyri is production ready.
The three stages are:

- As a library maintainer, generate and publish papyri IRD file.
- As a system operator, install IRD files on your system
- As a user, view the documentation in your IDE, webbrowser... with your preferences.


To generate IRD files::

  $ papyri gen <config file>

You can look in the ``examples`` folder to see some of these config files.
Those will put IRD files in ``~/.papyri/data`` there is no upload mechanism yet.

To install those files::

  $ papyri ingest ~/.papyri/data/library_version/


And finally to view the docs, either follow your IED documentation or use some of the
built-in renderer::


  $ papyri render

  $ papyri browse ....

  $ papyri serve


Of Interest
-----------

Here are a couple of function that are of interest to explore what papyri can do and render.

`dask.delayed`
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


Directive must not have spaces before double colon::

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
import json

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


""",
pretty_exceptions_enable=False
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
    relink: bool = False,
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
    check : bool
        <Multiline Description Here>
    dummy_progress : bool
        <Multiline Description Here>
    """
    _intro()
    from . import crosslink as cr

    for p in paths:
        cr.main(Path(p), check, dummy_progress=dummy_progress)
    if relink:
        cr.relink(dummy_progress=dummy_progress)


ROOT = "https://pydocs.github.io/pkg"


@app.command()
def install(
    names: List[str],
    check: bool = False,
    dummy_progress: bool = typer.Option(False, help="Disable rich progress bar"),
    relink: bool = False,
):
    """
    WIP, download and install a remote docbundle
    """

    from io import BytesIO
    from tempfile import TemporaryDirectory

    import httpx
    import rich
    import trio

    from . import crosslink as cr


    _intro()

    async def get(name, version, results, progress):
        """
        Utility to download a single docbundle and
        put it into result.

        """

        buf = BytesIO()

        client = httpx.AsyncClient()

        async with client.stream("GET", f"{ROOT}/{name}-{version}.zip") as response:
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
        client = httpx.AsyncClient()
        index = (await client.get(f"{ROOT}/index.json")).json()
        assert len(set(names)) == len(names)

        requested = {}
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
            requested[name] = version

        to_download = {}
        for k, v in requested.items():
            if k not in index["packages"]:
                print(f"No documentation found for {k!r}")
                continue
            if v not in index["packages"][k]:
                print(
                    f"Could not find {k}=={v}, available versions are {index['packages'][k]}"
                )
            to_download[k] = v

        with rich.progress.Progress(
            "{task.description}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            rich.progress.BarColumn(bar_width=None),
            rich.progress.DownloadColumn(),
            rich.progress.TransferSpeedColumn(),
        ) as progress:
            async with trio.open_nursery() as nursery:
                for name, version in to_download.items():
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
    if datas and relink:
        cr.relink(dummy_progress=dummy_progress)


@app.command()
def relink(
    dummy_progress: bool = typer.Option(False, help="Disable rich progress bar"),
):
    """
    Rescan all the documentation to find potential new crosslinks.
    """
    _intro()
    from . import crosslink as cr

    cr.relink(dummy_progress=dummy_progress)


@app.command()
def gen(
    files: List[str],
    infer: Optional[bool] = typer.Option(
        True, help="Whether to run type inference on code examples."
    ),
    exec: Optional[bool] = typer.Option(
        None, help="Whether to attempt to execute doctring code."
    ),
    debug: bool = False,
    dummy_progress: bool = typer.Option(False, help="Disable rich progress bar"),
    dry_run: bool = False,
    api: bool = True,
    examples: bool = True,
    narrative: bool = True,
    fail: bool = typer.Option(False, help="Fail on first error"),
    fail_early: bool = typer.Option(False, help="Overwrite early error option"),
    fail_unseen_error: bool = typer.Option(
        False, help="Overwrite fail on unseen error option"
    ),
):
    """
    Generate documentation for a given package.

    First item should be the root package to import, if subpackages need to be
    analyzed  but are not accessible from the root pass them as extra arguments.

    """
    _intro()
    from papyri.gen import gen_main
    from IPython.utils.tempdir import TemporaryWorkingDirectory

    if len(files) > 1:
        print(
            """
            Warning, it is not recommended to run papyri on multiple libraries at once,
            as many libraries might have side effects. """
        )
    from os.path import join
    import os

    here = os.getcwd()

    for file in files:
        with TemporaryWorkingDirectory():
            gen_main(
                infer=infer,
                exec_=exec,
                target_file=join(here, file),
                debug=debug,
                dummy_progress=dummy_progress,
                dry_run=dry_run,
                api=api,
                examples=examples,
                fail=fail,
                narrative=narrative,
                fail_early=fail_early,
                fail_unseen_error=fail_unseen_error,
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
    ascii: bool = False,
    html: bool = True,
    dry_run: bool = False,
    sidebar: bool = True,
    graph: bool = True,
    minify: bool = False,
):
    _intro()
    import trio

    from .render import main as m2

    trio.run(m2, ascii, html, dry_run, sidebar, graph, minify)


@app.command()
def ascii(name: str):
    _intro()
    import trio

    from .render import ascii_render

    trio.run(ascii_render, name)


@app.command()
def serve(sidebar: bool = True, port: int = 1234):
    _intro()
    from .render import serve as s2

    s2(sidebar=sidebar, port=port)


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

    RST = Language(spth, "rst")


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
