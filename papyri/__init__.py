"""
Papyri – in progress
"""
__version__ = "0.0.1"

import click

from . import gen as generate
from . import crosslink as cr
from . import render as rd
from .config import base_dir, logo


@click.group()
def main():
    print(logo) 
    pass


@click.command()
@click.argument("names", nargs=-1)
@click.option('--infer/--no-infer', default=True)
def gen(names, infer):
    generate.main(names, infer=infer)


@click.command()
def crosslink():
    cr.main()

@click.command()
def render():
    rd.main()

@click.command()
@click.argument('qualname', required=True)
def open(qualname):
    import webbrowser
    path = base_dir / 'html' / (qualname + '.html')
    if not path.exists():
        import sys
        sys.exit('No doc for '+ qualname+f" ({path})")
    print('opening', str(path))
    webbrowser.get().open('file://'+str(path), new=1)

main.add_command(gen)
main.add_command(crosslink)
main.add_command(render)
main.add_command(open)

if __name__ == "__main__":
    main()
