import os
import re
from pathlib import Path

import requests

from typing import List


class BaseStore:
    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        assert isinstance(path, Path)
        self.path = path

    def _other(self):
        return type(self)

    def __truediv__(self, other):
        if isinstance(other, str):
            return self._other()(self.path / other)
        elif isinstance(other, Store):
            raise ValueError

    def __str__(self):
        return str(self.path)

    async def exists(self):
        return self.path.exists()

    async def read_text(self):
        return self.path.read_text()

    def glob(self, arg) -> List[Path]:
        return [self._other()(x) for x in self.path.glob(arg)]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.path} 0x{id(self):x}>"

    @property
    def name(self):
        return self.path.name

    def __lt__(self, other):
        return self.path < other.path

    def __le__(self, other):
        return self.path <= other.path

    def __eq__(self, other):
        return self.path == other.path

    @property
    def parts(self):
        return self.path.parts


PAT = os.environ.get("PAT", None)
header = f"token {PAT}"


def gre(pat):
    return re.compile(pat.replace(".", r"\.").replace("*", ".+"))


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
