import json
import sqlite3
from collections import namedtuple
from pathlib import Path as _Path
from typing import List, Tuple


class Path:
    """just a path wrapper that has a conveninent `.read_json` and `.write_json` method"""

    def __init__(self, path):
        assert isinstance(path, _Path), path
        self.path = path

    def read_json(self):
        return json.loads(self.path.read_text())

    def write_json(self, data):
        self.path.write_text(json.dumps(data))

    def __truediv__(self, other):
        return type(self)(self.path / other)

    def write_bytes(self, *args, **kwargs):
        self.path.write_bytes(*args, **kwargs)

    @property
    def parent(self):
        return self.path.parent

    def exists(self, *args, **kwargs):
        return self.path.exists(*args, **kwargs)

    def mkdir(self, *args, **kwargs):
        self.path.mkdir(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.path, name)


Key = namedtuple("Key", ["module", "version", "kind", "path"])


class GraphStore:
    """
    Class abstraction over the filesystem to store documents in a graph-like
    structure

    We do not want to use the server or alike as most of our users will use REPL
    or IDE and performance is not our goal as we are likely going to access
    nodes infrequently.

    Plus we are not that much interested in the graph structure (yet), than the
    local neighborhood.

    We also have some specificities around document versions, which I'm not
    clear yet as to how to deal with, as well as dangling edges.

    Each document is stored with an ~4 item keys:

     - package it belongs to
     - version
     - kind of document
     - name of document

    Each document will also have references to others, with types; those are our
    edges.

    When we get a document, we do want to get as well all most recent the
    documents that reference it and why that is to say the coming edges.

    When we put a document, we ask for all the documents this references; and
    should update the edges accordingly.

    I don't really want to store a global edge document – though it might seem
    the most reasonable; sqlite ? So I was thinking of storing a companion
    document with all the back references; but maybe sqlite is a better
    approach.

    One more question is about the dangling documents? Like document we have references to,
    but do not exist yet, and a bunch of other stuff.

    """

    def __init__(self, root: _Path, link_finder=None):

        # for now we are going to try to do in-memory operation, just to
        # see how we can handle that with SQL, and move to on-disk later.
        p = _Path("~/.papyri/ingest/papyri.db")
        p = p.expanduser()
        if not p.exists():
            self.table = sqlite3.connect(str(p))
            print("Creating link table")
            self.table.cursor().execute(
                "CREATE TABLE links(source, dest, reason, unique(source, dest, reason))"
            )
        else:
            self.table = sqlite3.connect(str(p))

        # assert isinstance(link_finder, dict)
        assert isinstance(root, _Path)
        self._root = Path(root)
        self._link_finder = link_finder

    def _key_to_paths(self, key: Key) -> Tuple[Path, Path]:
        """
        Given A key, return path to the current file
        and the back referenced.

        Parameters
        ----------
        key : Key

        Returns
        -------
        data_path:  _Path
        backref_path : _Path

        """
        path = self._root
        assert None not in key, key
        for k in key[:-1]:
            path = path / k
        path0 = path / (key[-1])
        path_br = path / (key[-1] + ".br")
        return path0, path_br

    def _path_to_key(self, path: Path):
        """
        Given a path, return the key for the document.

        Opposite of _key_to_path

        Parameters
        ----------
        path : Path

        Returns
        -------
        key : Key
        """
        path = path.relative_to(self._root.path)
        if len(path.parts) == 4:
            a, b, c, d = path.parts
            return Key(a, b, c, d)
        else:
            return path.parts

    def remove(self, key: Key) -> None:
        data, backrefs = self._key_to_paths(key)
        data.unlink()
        #  this is likely incorrect if we want to deal with dangling links.
        backrefs.unlink()
        print("Removign link from table")
        self.table.execute(
            "delete from links where source=?",
            (str(key),),
        )

    def get(self, key: Key) -> bytes:
        assert isinstance(key, Key)
        path, _ = self._key_to_paths(key)

        ## Verification block
        path_br = _

        if path_br.path.exists():
            xx = path_br.read_json()
            backrefs = set([Key(*item) for item in xx])
        else:
            backrefs = set([])

        sql_backref_unparsed = self.table.execute(
            "select distinct source from links where dest=?", (str(key),)
        )

        sql_backrefs = {eval(s[0]) for s in sql_backref_unparsed}

        if not sql_backrefs == backrefs:
            print(f"Backreferences for {key} differ:")
            print(f"    there are {len(sql_backrefs)=}")
            print(f"    and {len(backrefs)=}")
            print(" + sql : ", sql_backrefs - backrefs)
            print(" - json:", backrefs - sql_backrefs)

        ## end verification block

        return path.read_bytes()

    def get_backref(self, key: Key):
        _, pathbr = self._key_to_paths(key)

        # print("getting backrefs from table")
        self.table.execute(
            "select source, reason from links where dest=?",
            (str(key),),
        )

        if pathbr.path.exists():
            return pathbr.read_json()
        else:
            return []

    def _add_edge(self, source: Key, dest: Key):
        """
        Add a backward edge from source to dest in dest br file.
        """
        _, p = self._key_to_paths(dest)
        if p.path.exists():
            data = set([tuple(x) for x in p.read_json()])
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            data = set()
        data.add(source)
        p.write_json(list(sorted(data)))

    def _remove_edge(self, source: Key, dest: Key):
        """
        Remove the edge from `source` to `dest`,
        that is to say, goes in to `dest` backrefs and remove it.
        """
        assert isinstance(source, Key)
        assert isinstance(dest, Key)
        _, p = self._key_to_paths(dest)
        if p.exists():
            data = set(Key(*x) for x in p.read_json())
            data.discard(source)
            p.write_json(list(sorted(data)))

    def put(self, key: Key, bytes_, refs) -> None:
        """
        Store object ``bytes``, as path ``key`` with the corresponding
        links to other objects.


        TODO: refs is forward refs, and we are updating backward believe
        """
        assert isinstance(key, Key)
        path, _ = self._key_to_paths(key)
        path.path.parent.mkdir(parents=True, exist_ok=True)

        if "assets" not in key and path.exists():
            __tmp = json.loads(path.read_bytes().decode())

            old_refs = {
                (b["module"], b["version"], b["kind"], b["path"])
                for b in __tmp.get("refs", [])
            }
        else:
            old_refs = set()

        path.write_bytes(bytes_)

        new_refs = set(refs)

        removed_refs = old_refs - new_refs
        added_refs = new_refs - old_refs

        #        if removed_refs or added_refs:
        #            print(key)
        #            for o in sorted(removed_refs):
        #                print("    -", o)
        #            for n in sorted(added_refs):
        #                print("    +", n)

        with self.table:
            for ref in added_refs:
                self._add_edge(key, ref)
                refkey = Key(*ref)
                self.table.execute(
                    "insert or ignore into links values (?,?,?)",
                    (str(key), str(refkey), "debug"),
                )
            for ref in removed_refs:
                refkey = Key(*ref)
                self._remove_edge(key, refkey)
                self.table.execute(
                    "delete from links where source=? and dest=? and reason=?",
                    (str(key), str(refkey), "debug"),
                )

    def glob(self, pattern) -> List[Key]:
        acc = ""
        for p in pattern:
            if p is None:
                acc += "/*"
            else:
                acc += "/" + p
        acc = acc[1:]
        try:
            res = [
                self._path_to_key(p)
                for p in self._root.glob(acc)
                if not p.name.endswith(".br")
            ]  # !!
        except Exception as e:
            raise type(e)("Acc:" + acc, pattern)
        return res
