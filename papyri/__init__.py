"""
Papyri – in progress
"""
__version__ = "0.0.1"

import click

logo = r"""
  ___                    _ 
 | _ \__ _ _ __ _  _ _ _(_)
 |  _/ _` | '_ \ || | '_| |
 |_| \__,_| .__/\_, |_| |_|
          |_|   |__/       
"""


@click.group()
def main():
    print(logo)
    print(__version__)


@click.command()
@click.argument("paths", nargs=-1)
@click.option("--check/--no-check", default=True)
def ingest(paths, check):
    from . import crosslink as cr
    from pathlib import Path

    for p in paths:
        cr.main(Path(p), check)


@click.command()
@click.argument("names", nargs=-1)
@click.option("--infer/--no-infer", default=True)
@click.option("--exec/--no-exec", default=False)
def gen(names, infer, exec):
    from papyri.gen import gen_main

    gen_main(names, infer=infer, exec_=exec)


@click.command()
def render():
    import trio

    from .render import main as m2

    trio.run(m2)


@click.command()
@click.argument("name", nargs=1)
def ascii(name):
    import trio

    from .render import ascii_render

    trio.run(ascii_render, name)


@click.command()
def serve():
    from .render import serve

    serve()


@click.command()
@click.argument("qualname", required=True)
def open(qualname):
    import webbrowser

    from .config import html_dir

    path = html_dir / (qualname + ".html")
    if not path.exists():
        import sys

        sys.exit("No doc for " + qualname + f" ({path})")
    print("opening", str(path))
    webbrowser.get().open("file://" + str(path), new=1)


main.add_command(ingest)
main.add_command(gen)
main.add_command(render)
main.add_command(open)
main.add_command(ascii)
main.add_command(serve)

if __name__ == "__main__":
    main()
