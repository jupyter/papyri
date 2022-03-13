"""
This module contains utilities to manipulate the documentation units,
usually trees, and update nodes.

"""

from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Set, Tuple

from .take2 import (
    Admonition,
    BlockDirective,
    BlockMath,
    Code2,
    Directive,
    Link,
    Math,
    Node,
    RefInfo,
    Token,
    Verbatim,
)

_cache: Dict[int, Tuple[Dict[str, RefInfo], FrozenSet[str]]] = {}


# @lru_cache(maxsize=100000)
def _build_resolver_cache(
    known_refs: FrozenSet[RefInfo],
) -> Tuple[Dict[str, RefInfo], FrozenSet[str]]:
    """
    Build resolver cached.

    Here we build two caches:

    1) a mapping from fully qualified names to refinfo objects.
    2) a set of all the keys we know about.

    Parameters
    ----------
    known_refs : (frozen) set of RefInfo

    Returns
    -------
    mapping:
        Mapping from path to a RefInfo, this allows to quickly compute
        what is the actual refinfo for a give path/qualname
    keyset:
        Frozenset of the map keys.

    """

    _map: Dict[str, List[RefInfo]] = defaultdict(lambda: [])
    assert isinstance(known_refs, frozenset)
    for k in known_refs:
        assert isinstance(k, RefInfo)
        _map[k.path].append(k)

    _m2: Dict[str, RefInfo] = {}
    for kk, v in _map.items():
        cand = list(sorted(v, key=lambda x: "" if x.version is None else x.version))
        assert len({c.module for c in cand}) == 1, cand
        _m2[kk] = cand[-1]

    return _m2, frozenset(_m2.keys())


@lru_cache
def root_start(root, refs):
    """
    Compute a subset of references that start with given root.
    """
    return frozenset(r for r in refs if r.startswith(root))


@lru_cache(10000)
def endswith(end, refs):
    """
    Compute as subset of references that ends with given root.
    """
    return frozenset(r for r in refs if r.endswith(end))


def resolve_(
    qa: str,
    known_refs: FrozenSet[RefInfo],
    local_refs: FrozenSet[str],
    ref: str,
    rev_aliases=None,
) -> RefInfo:
    """
    Given the current context (qa), and a str (ref), compute the RefInfo object.

    References are often relative based on the current context (which object you
    are currently in).

    Given this informations and all the local (same document) and global
    (same library/all libraries) references, compute the Reference Info object.

    Parameters
    ----------
    qa : str
        fully qualified path of the current object (.valueTODO: this will be weird for
        non object, like example).
    known_refs : list of RefInfo
        All the known objects we can refer to in current universe.
    local_refs : list of str
        All the current objects in current scope (same docstring).
    rev_aliases:
        Reverse alias map. As the import name of object may not be the
        fully qualified names, we may need a reverse alias map to resolve
        with respect to the import name.

    """

    # RefInfo(module, version, kind, path)
    # print('resolve', qa)
    hk = hash(known_refs)
    hash(local_refs)
    if rev_aliases is None:
        rev_aliases = {}
    if ref in rev_aliases:
        new_ref = rev_aliases[ref]
        # print(f'now looking for {new_ref} instead of {ref}')
        assert new_ref not in rev_aliases, "would loop...."
        # TODOlikely can drop rev_aliases here
        res = resolve_(qa, known_refs, local_refs, new_ref, rev_aliases)
        return res

    assert isinstance(ref, str), ref

    # TODO: LRU Cache seem to have speed problem here; and get slow while this should be just fine.
    # this seem to be due to the fact that even if the hash is the same this still needs to compare the objects, as
    # those may have been muted.
    if hk not in _cache:
        _cache[hk] = _build_resolver_cache(known_refs)

    # this is a mappign from the key to the most relevant
    # Refinfo to a document
    k_path_map: Dict[str, RefInfo]

    # hashable for cachign /optimisation.
    keyset: FrozenSet[str]

    k_path_map, keyset = _cache[hk]

    if ref.startswith("builtins."):
        return RefInfo(None, None, "missing", ref)
    if ref.startswith("str."):
        return RefInfo(None, None, "missing", ref)
    if ref in {"None", "False", "True"}:
        return RefInfo(None, None, "missing", ref)
    # here is sphinx logic.
    # https://www.sphinx-doc.org/en/master/_modules/sphinx/domains/python.html?highlight=tilde
    # tilda ~ hide the module name/class name
    # dot . search more specific first.
    if ref.startswith("~"):
        ref = ref[1:]
    if ref in local_refs:
        return RefInfo(None, None, "local", ref)
    if ref in k_path_map:
        # get the more recent.
        # stuff = {k for k in known_refs if k.path == ref}
        # c2 = list(sorted(stuff, key=lambda x: x.version))[-1]
        # assert isinstance(c2, RefInfo), c2
        # assert k_path_map[ref] == c2
        return k_path_map[ref]
    else:
        if ref.startswith("."):
            if (found := qa + ref) in k_path_map:
                return k_path_map[found]
            else:
                root = qa.split(".")[0]
                sub1 = root_start(root, keyset)
                subset = endswith(ref, sub1)
                if len(subset) == 1:
                    return k_path_map[next(iter(subset))]
                    # return RefInfo(None, None, "exists", next(iter(subset)))
                else:
                    if len(subset) > 1:
                        # ambiguous ref
                        # print("subset:", ref)
                        pass

                # print(f"did not resolve {qa} + {ref}")
                return RefInfo(None, None, "missing", ref)

        parts = qa.split(".")
        for i in range(len(parts)):
            attempt = ".".join(parts[:i]) + "." + ref
            if attempt in k_path_map:
                return k_path_map[attempt]

    q0 = qa.split(".")[0]
    rs = root_start(q0, keyset)
    attempts = [q for q in rs if (ref in q)]
    if len(attempts) == 1:
        # return RefInfo(None, None, "exists", attempts[0])
        return k_path_map[attempts[0]]
    else:
        trail = [q for q in attempts if q.split(".")[-1] == ref]
        if len(trail) == 1:
            return k_path_map[trail[0]]

    return RefInfo(None, None, "missing", ref)


class TreeVisitor:
    def __init__(self, find):
        self.skipped = set()
        self.find = find

    def generic_visit(self, node):
        name = node.__class__.__name__
        if method := getattr(self, "visit_" + name, None):
            return method(node)
        elif hasattr(node, "children"):
            acc = {}
            for c in node.children:
                if c is None or isinstance(c, (str, bool)):
                    continue
                assert c is not None, f"{node=} has a None child"
                assert isinstance(c, Node), repr(c)
                if type(c) in self.find:
                    acc.setdefault(type(c), []).append(c)
                else:
                    for k, v in self.generic_visit(c).items():
                        acc.setdefault(k, []).extend(v)
            return acc
        elif hasattr(node, "value"):
            if type(node) not in self.skipped:
                self.skipped.add(type(node))
            return {}
        else:
            raise ValueError(f"{node.__class__} has no children, no values {node}")


class TreeReplacer:
    """
    Tree visitor with methods to replace nodes.

    define replace_XXX(xxx) that return a list of new nodes, and call visit(and the root tree)
    """

    def __init__(self):
        self._replacements = Counter()

    def visit(self, node):
        self._replacements = Counter()
        assert not isinstance(node, list)
        res = self.generic_visit(node)
        assert len(res) == 1
        return res[0]

    def generic_visit(self, node) -> List[Node]:
        assert node is not None
        try:
            name = node.__class__.__name__
            if method := getattr(self, "replace_" + name, None):
                self._replacements.update([name])
                new_nodes = method(node)
            elif name in [
                "Word",
                "Verbatim",
                "Example",
                "BlockVerbatim",
                "Math",
                "Link",
                "Code",
                "Fig",
                "Words",
                "Comment",
                "BlockQuote",
                "BulletList",
                "Directive",
                "SeeAlsoItems",
                "Code2",
                "BlockMath",
                "Unimplemented",
                "Options",
                "SubstitutionRef",
            ]:
                return [node]
            elif name in ["Text"]:
                assert False, "Text still present"
            else:
                new_children = []
                if not hasattr(node, "children"):
                    raise ValueError(f"{node.__class__} has no children {node}")
                for c in node.children:
                    assert c is not None, f"{node=} has a None child"
                    assert isinstance(c, Node), c
                    replacement = self.generic_visit(c)
                    # if node.__class__.__name__ == "Param":
                    #    print(
                    #        "Param has children",
                    #        [x.__class__.__name__ for x in replacement],
                    #    )
                    assert isinstance(replacement, list)
                    new_children.extend(replacement)
                node.children = new_children
                new_nodes = [node]
            assert isinstance(new_nodes, list)
            return new_nodes
        except Exception as e:
            raise type(e)(f"{node=}")


class DirectiveVisiter(TreeReplacer):
    """
    A tree replacer to update directives.

    """

    def __init__(self, qa, known_refs: FrozenSet[RefInfo], local_refs, aliases):
        super().__init__()
        assert isinstance(qa, str), qa
        assert isinstance(known_refs, (list, set, frozenset)), known_refs
        self.known_refs = frozenset(known_refs)
        self.local_refs = frozenset(local_refs)
        self.qa = qa
        self.local: List[str] = []
        self.total: List[Tuple[Any, str]] = []
        # long -> short
        self.aliases: Dict[str, str] = aliases
        # short -> long
        self.rev_aliases = {v: k for k, v in aliases.items()}
        self._targets: Set[Any] = set()

    def replace_BlockDirective(self, block_directive: BlockDirective):
        block_directive.children = [self.visit(c) for c in block_directive.children]

        if block_directive.directive_name in [
            "versionchanged",
            "versionadded",
            "deprecated",
        ]:
            # TODO:
            if len(block_directive.args0) == 1:
                title, children = block_directive.args0[0], block_directive.children
            else:
                title, children = "TODO", block_directive.children

            return [
                Admonition(
                    block_directive.directive_name,
                    title,
                    children,
                )
            ]

        elif block_directive.directive_name in ["math"]:
            # assert len(block_directive.args0) == 1
            if not block_directive.children:
                assert len(block_directive.args0) == 1, (
                    block_directive.args0,
                    block_directive.children,
                )
            if ch := block_directive.children:
                assert len(ch) == 1
                assert not ch[0].inner
                res = BlockMath(
                    " ".join(block_directive.args0 + [w.value for w in ch[0].inline])
                )

            else:
                res = BlockMath(block_directive.args0[0])

            return [res]
        elif block_directive.directive_name in ["warning", "note"]:
            args0 = block_directive.args0
            args0 = [a.strip() for a in args0 if a.strip()]
            if args0:
                # assert len(args0) == 1
                # TODO: dont' allow admonition on first line.
                # print(
                #    "ADM!!",
                #    self.qa,
                #    "does title block adm",
                #    repr(args0),
                #    repr(block_directive.children),
                # )
                title = args0[0]
            else:
                title = ""

            assert block_directive.children is not None, block_directive
            return [
                Admonition(
                    block_directive.directive_name, title, block_directive.children
                )
            ]
        if block_directive.directive_name in [
            "code",
            "autosummary",
            "note",
            "warning",
            "attribute",
            "hint",
            "plot",
            "seealso",
            "moduleauthor",
            "data",
            "WARNING",
            "currentmodule",
            "important",
            "code-block",
            "image",
            "rubric",
            "inheritance-diagram",
            "table",
        ]:
            # print("TODO:", block_directive.directive_name)
            return [block_directive]
        return [block_directive]

    def _resolve(self, loc, text):
        """
        Resolve `text` within local references `loc`

        """
        assert isinstance(text, str)
        return resolve_(
            self.qa, self.known_refs, loc, text, rev_aliases=self.rev_aliases
        )

    def replace_Directive(self, directive: Directive):
        if (directive.domain, directive.role) == ("py", "func"):
            pass
        elif (directive.domain, directive.role) == (None, None) and directive.value in (
            # TODO: link to stdlib
            "None",
            "True",
            "False",
        ):
            return [Verbatim([directive.value])]
        elif (directive.domain is not None) or (
            directive.role not in (None, "mod", "class", "func", "meth", "any")
        ):
            # TODO :many of these directive need to be implemented
            if directive.role == "math":
                m = Math([directive.value])
                return [m]
            if directive.role not in (
                "attr",
                "meth",
                "doc",
                "ref",
                "func",
                "mod",
                "class",
                "term",
                "exc",
                "obj",
                "data",
                "sub",
                "program",
                "file",
                "command",
                "sup",
                "samp",  # networkx
                "rc",  # matplotlib
            ):
                print("TODO role:", directive.role)
            return [directive]
        loc: FrozenSet[str]
        if directive.role not in ["any", None]:
            loc = frozenset()
        else:
            loc = self.local_refs
        text = directive.value
        # TODO: wrong, there should not be any ` left that is likely a
        # verbatim vs directive parsing issue.
        assert "`" not in text
        # text = text.strip("`")
        to_resolve = text
        if " <" in text and text.endswith(">"):
            try:
                text, to_resolve = text.split(" <")
            except ValueError:
                assert False, directive.value
            assert to_resolve.endswith(">"), (text, to_resolve)
            to_resolve = to_resolve.rstrip(">")

        r = self._resolve(loc, to_resolve)
        # this is now likely incorrect as Ref kind should not be exists,
        # but things like "local", "api", "gallery..."
        ref, exists = r.path, r.kind
        if exists != "missing":
            if exists == "local":
                self.local.append(text)
            else:
                self.total.append((text, ref))
            if r.kind != "local":
                assert None not in r, r
                self._targets.add(r)
            return [Link(text, r, exists, exists != "missing")]
        return [directive]


class DVR(DirectiveVisiter):
    def __init__(self, *args, version="??", **kwargs):
        self.version = version
        assert version != "??"
        assert version != ""
        super().__init__(*args, **kwargs)

    def replace_Code2(self, code):
        new_entries = []
        for token in code.entries:
            # TODO
            if isinstance(token.link, str):
                r = self._resolve(frozenset(), token.link)
                if r.kind == "module":
                    self._targets.add(r)
                    new_entries.append(
                        Token(
                            Link(
                                token.link,
                                r,
                                "module",
                                True,
                            ),
                            token.type,
                        )
                    )
                    continue
            new_entries.append(token)

        return [Code2(new_entries, code.out, code.ce_status)]

    def replace_Code(self, code):
        """
        Here we'll crawl example data and convert code entries so that each token contain a link to the object they
        refered to.
        """
        # TODO: here we'll have a problem as we will love the content of entry[1]. This should really be resolved at gen
        # time.
        new_entries = []
        for entry in code.entries:
            # TODO
            if entry[1] and entry[1].strip():
                r = self._resolve(frozenset(), entry[1])
                if r.kind == "module":
                    self._targets.add(r)
                    new_entries.append(
                        Token(
                            Link(
                                str(entry[0]),
                                r,
                                "module",
                                True,
                            ),
                            entry[2],
                        )
                    )
                    continue
            new_entries.append(
                Token(str(entry[0]), entry[2]),
            )

        return [Code2(new_entries, code.out, code.ce_status)]

    def replace_Fig(self, fig):

        # todo: add version number here
        self._targets.add(
            RefInfo(self.qa.split(".")[0], self.version, "assets", fig.value)
        )

        return [fig]
