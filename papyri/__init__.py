"""
Papyri – in progress
"""
__version__ = "0.0.1"

import click

logo = """\
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
@click.argument("names", nargs=-1)
@click.option("--infer/--no-infer", default=True)
def gen(names, infer):
    from . import gen as generate
    generate.main(names, infer=infer)


@click.command()
@click.option("--check/--no-check", default=True)
def ingest(check):
    from . import crosslink as cr
    cr.main(check)


@click.command()
def render():
    from .render import main
    main()


@click.command()
def serve():
    from .render import serve
    rd.serve()


@click.command()
@click.argument("qualname", required=True)
def open(qualname):
    import webbrowser
    from .config import html_dir
    path = html_dir / "html" / (qualname + ".html")
    if not path.exists():
        import sys

        sys.exit("No doc for " + qualname + f" ({path})")
    print("opening", str(path))
    webbrowser.get().open("file://" + str(path), new=1)


main.add_command(gen)
main.add_command(ingest)
main.add_command(render)
main.add_command(open)
main.add_command(serve)

if __name__ == "__main__":
    main()
