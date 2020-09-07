import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from quart_trio import QuartTrio
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape
from velin import NumpyDocString

from numpydoc.docscrape import Parameter

from .config import base_dir, html_dir, ingest_dir
from .crosslink import SeeAlsoItem, load_one, resolve_
from .take2 import Lines, Paragraph, make_block_3
from .utils import progress

import requests

import os

PAT = os.environ.get("PAT", None)


# maybe from cachetools import TTLCache


class BaseStore:
    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        assert isinstance(path, Path)
        self.path = path

    def _other(self):
        return type(self)

    def __truediv__(self, other):
        return self._other()(self.path / other)

    def __str__(self):
        return str(self.path)

    async def exists(self):
        return self.path.exists()

    async def read_text(self):
        return self.path.read_text()

    def glob(self, arg):
        return [self._other()(x) for x in self.path.glob(arg)]

    @property
    def name(self):
        return self.path.name

    def __lt__(self, other):
        return self.path < other.path

    def __le__(self, other):
        return self.path <= other.path

    def __eq__(self, other):
        return self.path == other.path


import re


def gre(pat):
    return re.compile(pat.replace(".", "\.").replace("*", ".+"))


header = f"token {PAT}"


class RCache:
    def __init__(self):
        from cachetools import TTLCache

        self.c = TTLCache(1024, 60)

    async def aget(self, url, headers=None):
        self.c.expire()
        if not (res := self.c.get(url)):
            res = requests.get(url, headers=headers)
            self.c[url] = res
        return res

    def get(self, url, headers=None):
        self.c.expire()
        if not (res := self.c.get(url)):
            res = requests.get(url, headers=headers)
            self.c[url] = res
        return res


RC = RCache()


class GHStore(BaseStore):
    def _other(self):
        return lambda p: type(self)(p, self.acc)

    def __init__(self, path, acc=None):
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path
        self.acc = acc

    def glob(self, arg):
        data = RC.get(
            "https://api.github.com/repos/Carreau/papyri-data/git/trees/master",
            headers={"Authorization": header},
        ).json()
        r = gre(arg)
        res = []
        for item in data["tree"]:
            data = RC.get(item["url"], headers={"Authorization": header}).json()
            res += [
                self._other()(Path(k)) for x in data["tree"] if r.match(k := x["path"])
            ]

        return res

    async def read_text(self):
        data = (
            await RC.aget(
                f"https://api.github.com/repos/Carreau/papyri-data/contents/ingest/{str(self.path)}",
                headers={"Authorization": header},
            )
        ).json()
        raw = await RC.aget(data["download_url"])
        return raw.text

    async def exists(self):
        data = (
            await RC.aget(
                f"https://api.github.com/repos/Carreau/papyri-data/contents/ingest/{str(self.path)}",
                headers={"Authorization": header},
            )
        ).json()
        res = "download_url" in data
        return res


class Store(BaseStore):
    pass


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


async def _route(ref, store):
    assert isinstance(store, BaseStore)
    if ref == "favicon.ico":
        here = Path(os.path.dirname(__file__))
        return (here / ref).read_bytes()

    if ref.endswith(".html"):
        ref = ref[:-5]
    if ref == "favicon.ico":
        return ""

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    env.globals["paragraphs"] = paragraphs
    template = env.get_template("core.tpl.j2")

    error = env.get_template("404.tpl.j2")
    known_refs = [str(x.name)[:-5] for x in store.glob("*.json")]

    file_ = store / f"{ref}.json"

    family = sorted(list(store.glob("*.json")))
    family = [str(f.name)[:-5] for f in family]
    parts = ref.split(".") + ["..."]
    siblings = {}
    cpath = ""
    # TODO: move this at ingestion time for all the non-top-level.
    for i, part in enumerate(parts):
        sib = list(
            sorted(
                set(
                    [
                        ".".join(s.split(".")[: i + 1])
                        for s in family
                        if s.startswith(cpath)
                    ]
                )
            )
        )
        cpath += part + "."

        siblings[part] = [(s, s.split(".")[-1]) for s in sib]

    if await file_.exists():
        bytes_ = await ((store / f"{ref}.json").read_text())
        brpath = store / f"{ref}.br"
        if await brpath.exists():
            br = await brpath.read_text()
        else:
            br = None
        ndoc = load_one(bytes_, br)
        local_ref = [x[0] for x in ndoc["Parameters"] if x[0]] + [
            x[0] for x in ndoc["Returns"] if x[0]
        ]
        env.globals["resolve"] = resolve_(ref, known_refs, local_ref)

        return render_one(template=template, ndoc=ndoc, qa=ref, ext="", parts=siblings)
    else:
        known_refs = [str(s.name)[:-5] for s in store.glob(f"{ref}*.json")]
        brpath = store / "__phantom__" / f"{ref}.json"
        if await brpath.exists():
            br = json.loads(await brpath.read_text())
        else:
            br = []
        return error.render(subs=known_refs, backrefs=list(set(br)))


def img(subpath):
    assert subpath.endswith("png")
    with open(ingest_dir / subpath, "rb") as f:
        return f.read()


def serve():
    import os

    app = QuartTrio(__name__)

    async def r(ref):
        return await _route(ref, Store(str(ingest_dir)))

    # return await _route(ref, GHStore(Path('.')))

    app.route("/<ref>")(r)
    app.route("/img/<path:subpath>")(img)
    port = os.environ.get("PORT", 5000)
    print("Seen config port ", port)
    prod = os.environ.get("PROD", None)
    if prod:
        app.run(port=port, host="0.0.0.0")
    else:
        app.run(port=port)


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


def render_one(template, ndoc, qa, ext, parts={}):
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
        parts=parts,
    )


async def _ascii_render(name, store=Store(ingest_dir)):
    ref = name

    env = Environment(
        loader=CleanLoader(os.path.dirname(__file__)),
        lstrip_blocks=True,
        trim_blocks=True,
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    env.globals["paragraphs"] = paragraphs
    template = env.get_template("ascii.tpl.j2")

    known_ref = [x.name[:-5] for x in store.glob("*")]
    bytes_ = await (store / f"{ref}.json").read_text()
    brpath = store / f"{ref}.br"
    if await brpath.exists():
        br = await brpath.read_text()
    else:
        br = None
    ndoc = load_one(bytes_, br)
    local_ref = [x[0] for x in ndoc["Parameters"] if x[0]] + [
        x[0] for x in ndoc["Returns"] if x[0]
    ]

    env.globals["resolve"] = resolve_(ref, known_ref, local_ref)

    return render_one(template=template, ndoc=ndoc, qa=ref, ext="")


async def ascii_render(*args, **kwargs):
    print(await _ascii_render(*args, **kwargs))


async def main():
    # nvisited_items = {}
    store = Store(ingest_dir)
    files = store.glob("*")

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(__file__)),
        autoescape=select_autoescape(["html", "tpl.j2"]),
    )
    env.globals["len"] = len
    env.globals["paragraph"] = paragraph
    template = env.get_template("core.tpl.j2")

    known_ref = [x.name[:-5] for x in store.glob("*")]

    html_dir.mkdir(exist_ok=True)
    for p, fname in progress(files, description="Rendering..."):
        if fname.startswith("__") or fname.endswith(".br"):
            continue
        qa = fname[:-5]
        try:
            bytes_ = await (store / fname).read_text()
            brpath = store / f"{qa}.br"
            if await brpath.exists():
                br = await brpath.read_text()
            else:
                br = None
            ndoc = load_one(bytes_, br)

            local_ref = [x[0] for x in ndoc["Parameters"] if x[0]]
            # nvisited_items[qa] = ndoc
        except Exception as e:
            raise RuntimeError(f"error with {fname}") from e

        # for p,(qa, ndoc) in progress(nvisited_items.items(), description='Rendering'):
        env.globals["resolve"] = resolve_(qa, known_ref, local_ref)
        with (html_dir / f"{qa}.html").open("w") as f:
            f.write(render_one(template=template, ndoc=ndoc, qa=qa, ext=".html"))
