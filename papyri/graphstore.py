# import json
import cbor2
import sqlite3
from collections import namedtuple
from pathlib import Path as _Path
from typing import List, Set


class Path:
    """just a path wrapper that has a conveninent `.read_json` and `.write_json` method"""

    def __init__(self, path):
        assert isinstance(path, _Path), path
        self.path = path

    def read_json(self):
        with open(self.path, "rb") as f:
            return cbor2.load(f)
            # return json.loads(self.path.read_text())

    def write_json(self, data):
        with open(self.path, "wb") as f:
            return cbor2.dump(data, f)
        # self.path.write_text(json.dumps(data))

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
            self.conn = sqlite3.connect(str(p))
            self.conn.execute("PRAGMA foreign_keys = 1")

            print("Creating documents table")
            self.conn.cursor().execute(
                """
                CREATE TABLE documents(
                id INTEGER PRIMARY KEY,
                package TEXT NOT NULL,
                version TEXT NOT NULL,
                category TEXT NOT NULL,
                identifier TEXT NOT NULL, unique(package, version, category, identifier))
                """
            )

            self.conn.cursor().execute(
                """
                CREATE TABLE destinations(
                id INTEGER PRIMARY KEY,
                package TEXT NOT NULL,
                version TEXT NOT NULL,
                category TEXT NOT NULL,
                identifier TEXT NOT NULL, unique(package, version, category, identifier))
                """
            )

            print("Creating links table")
            self.conn.cursor().execute(
                """
                CREATE TABLE links(
                id INTEGER PRIMARY KEY,
                source INTEGER NOT NULL,
                dest INTEGER NOT NULL,
                metadata TEXT,
                FOREIGN KEY (source) REFERENCES documents(id) ON DELETE CASCADE
                FOREIGN KEY (dest) REFERENCES destinations(id) ON DELETE CASCADE)
                """
            )
            self.conn.cursor().execute(
                """
                CREATE INDEX module on documents(package) ;
                CREATE INDEX px on documents(identifier);
                CREATE INDEX qa on destinations(identifier);
                CREATE INDEX ax on destinations(package, version, category, identifier);
                CREATE INDEX sx on links(source);
                CREATE INDEX dx on links(dest);
                """
            )

            self.conn.commit()
        else:
            self.conn = sqlite3.connect(str(p))

        # assert isinstance(link_finder, dict)
        assert isinstance(root, _Path)
        self._root = Path(root)
        self._link_finder = link_finder

    def _key_to_path(self, key: Key) -> Path:
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
        return path0

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
        path = self._key_to_path(key)
        path.unlink()
        #  this is likely incorrect if we want to deal with dangling links.
        print("Removing link from table")
        self.conn.execute(
            "delete from links where source=?",
            (str(key),),
        )

    def _get(self, key: Key) -> bytes:
        assert isinstance(key, Key)
        path = self._key_to_path(key)

        # TODO: this is partially incorrect.
        # as we only match on the identifier,
        # we should match on more.
        return path.read_bytes()

    def _get_backrefs(self, key: Key) -> Set[Key]:
        cur = self.conn.cursor()
        backrows = list(
            cur.execute(
                """
        select documents.*
        from links
            inner join documents on links.source=documents.id
            inner join destinations on links.dest=destinations.id
        where destinations.identifier=?""",
                (key.path,),
            )
        )

        sql_backrefs = {Key(*s[1:]) for s in backrows}
        return sql_backrefs

    def _get_forwardrefs(self, key: Key):
        cur = self.conn.cursor()
        forward_rows = list(
            cur.execute(
                """
        select destinations.*
        from links
            inner join documents on links.source=documents.id
            inner join destinations on links.dest=destinations.id
        where documents.identifier=?""",
                (key.path,),
            )
        )

        sql_forward_ref = {Key(*s[1:]) for s in forward_rows}
        return sql_forward_ref

    def get_all(self, key):
        return self._get(key), self._get_backrefs(key), self._get_forwardrefs(key)

    def get_backref(self, key: Key) -> Set[Key]:
        return self._get_backrefs(key)

    def get_forwardref(self, key: Key) -> Set[Key]:
        return self._get_forwardrefs(key)

    def get(self, key: Key) -> bytes:
        return self._get(key)

    def _maybe_insert_source(self, key):
        with self.conn:
            c1 = self.conn.cursor()
            rows = list(
                c1.execute(
                    """
                select id from documents where (
                    package=?
                AND version=?
                AND category=?
                AND identifier=?)
                """,
                    list(key),
                )
            )
            if not rows:
                c1.execute(
                    """
                    insert into documents values
                    (Null, ?, ?, ?, ?)
                    """,
                    list(key),
                )
                source_id = c1.lastrowid
            else:
                [(source_id,)] = rows

        return source_id

    def _maybe_insert_dest(self, ref):
        with self.conn:
            c1 = self.conn.cursor()
            rows = list(
                c1.execute(
                    """
                select id from destinations where (
                    package=?
                AND version=?
                AND category=?
                AND identifier=?)
                """,
                    list(ref),
                )
            )
            if not rows:
                c1.execute(
                    """
                    insert into destinations values
                    (Null, ?, ?, ?, ?)
                    """,
                    list(ref),
                )
                dest_id = c1.lastrowid
            else:
                [(dest_id,)] = rows

        return dest_id

    def put(self, key: Key, bytes_, refs) -> None:
        """
        Store object ``bytes``, as path ``key`` with the corresponding
        links to other objects.

        refs : List[Key] ?

        """
        assert isinstance(key, Key)
        for r in refs:
            assert isinstance(r, tuple), r
            assert len(r) == 4
        path = self._key_to_path(key)
        path.path.parent.mkdir(parents=True, exist_ok=True)

        if "assets" not in key and path.exists():
            old_refs = self.get_forwardref(key)
        else:
            old_refs = set()

        path.write_bytes(bytes_)

        new_refs = set(refs)
        del refs

        removed_refs = old_refs - new_refs
        added_refs = new_refs - old_refs

        with self.conn:
            source_id = self._maybe_insert_source(key)
            params = []
            for ref in added_refs:
                params.append((source_id, self._maybe_insert_dest(ref), "debug"))

            to_del = []
            for ref in removed_refs:
                to_del.append((source_id, self._maybe_insert_dest(ref)))
            c3 = self.conn.cursor()
            c3.executemany("insert or ignore into links values (NULL, ?,?,?)", params)
            c3.executemany("delete from links where source=? and dest=? ", to_del)

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
