import json
import os
from collections import defaultdict
from functools import lru_cache
from types import ModuleType

from flask import Flask
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape
from velin import NumpyDocString

from numpydoc.docscrape import Parameter

from .config import base_dir, html_dir, ingest_dir
from .crosslink import SeeAlsoItem, resolve_, load_one
from .take2 import Paragraph
from .utils import progress

app = Flask(__name__)


class CleanLoader(FileSystemLoader):


    def get_source(self, *args, **kwargs):
        (source, filename, uptodate) = super().get_source(*args, **kwargs)
        return until_ruler(source), filename, uptodate


def until_ruler(doc):
    """
    Utilities to clean jinja template; 
    
    Remove all ``|`` and `` `` until the last leading ``|``
    
    """
    lines = doc.split('\n')
    new = []
    for l in lines:
        
        while len(l.lstrip()) >= 1 and l.lstrip()[0] == '|':
            l = l.lstrip()[1:]
        new.append(l)
    return '\n'.join(new)


@app.route("/<ref>")
def route(ref):
    if ref.endswith(".html"):
        ref = ref[:-5]
    if ref == "favicon.ico":
        return ""

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["exists"] = exists
    env.globals["paragraph"] = paragraph
    template = env.get_template("core.tpl.j2")

    known_ref = [x.name[:-5] for x in ingest_dir.glob("*")]
    with open(ingest_dir / f"{ref}.json") as f:
        bytes_ = f.read()
    ndoc = load_one(bytes_)
    local_ref = [x[0] for x in ndoc["Parameters"] if x[0]]+[x[0] for x in ndoc["Returns"] if x[0]]

    env.globals["resolve"] = resolve_(ref, known_ref, local_ref)

    return render_one(template=template, ndoc=ndoc, qa=ref, ext="")


def serve():
    app.run()


def paragraph(lines):
    p = Paragraph.parse_lines(lines)
    acc = []
    for c in p.children:
        if type(c).__name__ == "Directive":
            if c.role == "math":
                acc.append(("Math", c))
            else:
                acc.append((type(c).__name__, c))
        else:
            acc.append((type(c).__name__, c))
    return acc


def render_one(template, ndoc, qa, ext):
    br = ndoc.backrefs
    if len(br) > 30:

        b2 = defaultdict(lambda: [])
        for ref in br:
            mod, _ = ref.split(".", maxsplit=1)
            b2[mod].append(ref)
        backrefs = (None, b2)
    else:
        backrefs = (br, None)
    return template.render(
        doc=ndoc,
        qa=qa,
        version=ndoc.version,
        module=qa.split(".")[0],
        backrefs=backrefs,
        ext=ext,
    )


#def load_one(bytes_):
#    data = json.loads(bytes_)
#    blob = NumpyDocString("")
#    blob._parsed_data = data.pop("_parsed_data")
#    blob._parsed_data["Parameters"] = [
#        Parameter(a, b, c) for (a, b, c) in blob._parsed_data["Parameters"]
#    ]
#    blob.refs = data.pop("refs")
#    blob.edata = data.pop("edata")
#    blob.backrefs = data.pop("backrefs",[])
#    blob.see_also = [SeeAlsoItem.from_json(**x) for x in data.pop("see_also", [])]
#    blob.__dict__.update(data)
#    return blob


@lru_cache()
def exists(ref):

    if (ingest_dir / f"{ref}.json").exists():
        return "exists"
    else:
        # if not ref.startswith(("builtins.", "__main__")):
        #    print(ref, "missing in", qa)
        return "missing"
    
def ascii_render(name):
    ref = name

    env = Environment(
        loader=CleanLoader(os.path.dirname(__file__)),
        lstrip_blocks=True,
        trim_blocks=True,
    )
    env.globals["exists"] = exists
    env.globals["paragraph"] = paragraph
    template = env.get_template("ascii.tpl.j2")

    known_ref = [x.name[:-5] for x in ingest_dir.glob("*")]
    with open(ingest_dir / f"{ref}.json") as f:
        bytes_ = f.read()
    ndoc = load_one(bytes_)
    local_ref = [x[0] for x in ndoc["Parameters"] if x[0]]+[x[0] for x in ndoc["Returns"] if x[0]]

    env.globals["resolve"] = resolve_(ref, known_ref, local_ref)

    print(render_one(template=template, ndoc=ndoc, qa=ref, ext=""))

def main():
    # nvisited_items = {}
    files = os.listdir(ingest_dir)

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["exists"] = exists
    env.globals["paragraph"] = paragraph
    template = env.get_template("core.tpl.j2")

    known_ref = [x.name[:-5] for x in ingest_dir.glob("*")]

    html_dir.mkdir(exist_ok=True)
    for p, fname in progress(files, description="Rendering..."):
        qa = fname[:-5]
        try:
            with open(ingest_dir / fname) as f:
                bytes_ = f.read()
                ndoc = load_one(bytes_)
                local_ref = [x[0] for x in ndoc["Parameters"] if x[0]]
                # nvisited_items[qa] = ndoc
        except Exception as e:
            raise RuntimeError(f"error with {f}") from e

        # for p,(qa, ndoc) in progress(nvisited_items.items(), description='Rendering'):
        with (html_dir / f"{qa}.html").open("w") as f:

            env.globals["resolve"] = resolve_(qa, known_ref, local_ref)

            f.write(render_one(template=template, ndoc=ndoc, qa=qa, ext=".html"))
