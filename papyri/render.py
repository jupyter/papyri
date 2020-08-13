import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from flask import Flask
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape
from numpydoc.docscrape import Parameter
from velin import NumpyDocString

from .config import base_dir, html_dir, ingest_dir
from .crosslink import SeeAlsoItem, load_one, resolve_
from .take2 import Lines, Paragraph, lines, make_block_3
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
    lines = doc.split("\n")
    new = []
    for l in lines:

        while len(l.lstrip()) >= 1 and l.lstrip()[0] == "|":
            l = l.lstrip()[1:]
        new.append(l)
    return "\n".join(new)


def _route(ref, ingest_dir):
    if ref.endswith(".html"):
        ref = ref[:-5]
    if ref == "favicon.ico":
        return ""

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["exists"] = exists
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    env.globals["paragraphs"] = paragraphs
    template = env.get_template("core.tpl.j2")

    error = env.get_template("404.tpl.j2")

    known_ref = [x.name[:-5] for x in ingest_dir.glob("*")]

    file_ = ingest_dir / f"{ref}.json"
    if file_.exists():
        with open(ingest_dir / f"{ref}.json") as f:
            bytes_ = f.read()
        brpath = Path(ingest_dir / f"{ref}.br")
        if brpath.exists():
            br = brpath.read_text()
        else:
            br = None
        ndoc = load_one(bytes_, br)
        local_ref = [x[0] for x in ndoc["Parameters"] if x[0]] + [
            x[0] for x in ndoc["Returns"] if x[0]
        ]

        env.globals["resolve"] = resolve_(ref, known_ref, local_ref)

        return render_one(template=template, ndoc=ndoc, qa=ref, ext="")
    else:
        known_refs = [str(s.name)[:-5] for s in ingest_dir.glob(f"{ref}*.json")]
        brpath = Path(ingest_dir / "__phantom__" / f"{ref}.json")
        if brpath.exists():
            br = json.loads(brpath.read_text())
        else:
            br = []
        print("br:", br, type(br))
        return error.render(subs=known_refs, backrefs=list(set(br)))


app.route("/<ref>")(lambda ref: _route(ref, ingest_dir))


def serve():
    app.run()


def paragraph(lines):
    """
    return container of (type, obj)
    """
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


def paragraphs(lines):
    blocks = make_block_3(Lines(lines))
    acc = []
    for b0, b1, b2 in blocks:
        if b0:
            acc.append(paragraph([x._line for x in b0]))
        ## definitively wrong but will do for now, should likely be verbatim, or recurse ?
        if b2:
            acc.append(paragraph([x._line for x in b2]))
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


@lru_cache()
def exists(ref):

    if (ingest_dir / f"{ref}.json").exists():
        return "exists"
    else:
        # if not ref.startswith(("builtins.", "__main__")):
        #    print(ref, "missing in", qa)
        return "missing"


def _ascii_render(name, ingest_dir=ingest_dir):
    ref = name

    env = Environment(
        loader=CleanLoader(os.path.dirname(__file__)),
        lstrip_blocks=True,
        trim_blocks=True,
    )
    env.globals["exists"] = exists
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    env.globals["paragraphs"] = paragraphs
    template = env.get_template("ascii.tpl.j2")

    known_ref = [x.name[:-5] for x in ingest_dir.glob("*")]
    with open(ingest_dir / f"{ref}.json") as f:
        bytes_ = f.read()
    brpath = Path(ingest_dir / f"{ref}.br")
    if brpath.exists():
        br = brpath.read_text()
    else:
        br = None
    ndoc = load_one(bytes_, br)
    local_ref = [x[0] for x in ndoc["Parameters"] if x[0]] + [
        x[0] for x in ndoc["Returns"] if x[0]
    ]

    env.globals["resolve"] = resolve_(ref, known_ref, local_ref)

    return render_one(template=template, ndoc=ndoc, qa=ref, ext="")


def ascii_render(*args, **kwargs):
    print(_ascii_render(*args, **kwargs))


def main():
    # nvisited_items = {}
    files = os.listdir(ingest_dir)

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["exists"] = exists
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    template = env.get_template("core.tpl.j2")

    known_ref = [x.name[:-5] for x in ingest_dir.glob("*")]

    html_dir.mkdir(exist_ok=True)
    for p, fname in progress(files, description="Rendering..."):
        if fname.startswith("__") or fname.endswith(".br"):
            continue
        qa = fname[:-5]
        try:
            with open(ingest_dir / fname) as f:
                bytes_ = f.read()
                brpath = Path(ingest_dir / f"{qa}.br")
                if brpath.exists():
                    br = brpath.read_text()
                else:
                    br = None
                ndoc = load_one(bytes_, br)

                local_ref = [x[0] for x in ndoc["Parameters"] if x[0]]
                # nvisited_items[qa] = ndoc
        except Exception as e:
            raise RuntimeError(f"error with {fname}") from e

        # for p,(qa, ndoc) in progress(nvisited_items.items(), description='Rendering'):
        with (html_dir / f"{qa}.html").open("w") as f:

            env.globals["resolve"] = resolve_(qa, known_ref, local_ref)

            f.write(render_one(template=template, ndoc=ndoc, qa=qa, ext=".html"))
