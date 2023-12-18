"""
This module contains utilities to manipulate the documentation units,
usually trees, and update nodes.

"""

import logging

from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Set, Tuple, Callable

from .take2 import (
    Directive,
    Link,
    RefInfo,
    SubstitutionDef,
)
from .common_ast import Node
from .myst_ast import (
    MMystDirective,
    MLink,
    MText,
    MList,
    MImage,
    MListItem,
    MInlineMath,
    MInlineCode,
    MCode,
    MParagraph,
    ReplaceNode,
    UnprocessedDirective,
)
from .utils import full_qual, FullQual, Cannonical, obj_from_qualname
from textwrap import indent
from .directives import (
    block_math_handler,
    note_handler,
    versionadded_handler,
    versionchanged_handler,
    deprecated_handler,
    warning_handler,
)

log = logging.getLogger("papyri")


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


class DelayedResolver:
    _targets: Dict[str, RefInfo]
    _references: Dict[str, List[Link]]

    def __init__(self):
        self._targets = dict()
        self._references = dict()

    def add_target(self, target_ref: RefInfo, target: str):
        assert target is not None
        assert target not in self._targets, "two targets with the same name"
        self._targets[target] = target_ref
        self._resolve(target)

    def add_reference(self, link: Link, target: str) -> None:
        self._references.setdefault(target, []).append(link)
        self._resolve(target)

    def _resolve(self, target: str) -> None:
        if (target in self._targets) and (target in self._references):
            for link in self._references[target]:
                link.reference = self._targets[target]
                # print("Updating link to point to", self._targets[target], link)
            self._references[target] = []


RESOLVER = DelayedResolver()


def resolve_(
    qa: str,
    known_refs: FrozenSet[RefInfo],
    local_refs: FrozenSet[str],
    ref: str,
    rev_aliases: Dict[Cannonical, FullQual],
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
    ref : str
        ???
    rev_aliases
        Reverse alias map. As the import name of object may not be the
        fully qualified names, we may need a reverse alias map to resolve
        with respect to the import name.

    """

    # RefInfo(module, version, kind, path)
    # print('resolve', qa)
    hk = hash(known_refs)
    hash(local_refs)
    assert rev_aliases is not None
    ref = Cannonical(ref)
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
    """
    Checks or modifies nodes in place.
    """

    def __init__(self, find):
        self.skipped = set()
        self.find = find

    def generic_visit(self, node):
        from .take2 import Options
        from .myst_ast import MThematicBreak

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
        elif hasattr(node, "reference"):
            acc = {}
            for c in [node.reference]:
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
        elif isinstance(node, (RefInfo, Options, MThematicBreak, SubstitutionDef)):
            return {}
        else:
            raise ValueError(f"{node.__class__} has no children, no values {node}")


class TreeReplacer:
    """
    Tree visitor with methods to replace nodes.

    define replace_XXX(xxx) that return a list of new nodes, and call visit(and the root tree)
    """

    _replacements: Counter

    def __init__(self):
        self._replacements = Counter()

    def visit(self, node):
        self._replacements = Counter()
        self._cr = 0
        assert not isinstance(node, list), node
        assert node is not None
        res = self.generic_visit(node)
        assert len(res) == 1, res
        return res[0]

    def _call_method(self, method, node):
        return method(node)

    def generic_visit(self, node) -> List[Node]:
        """
        Detects node type and dispatches to "visit_<type>" or "replace_<type>"
        methods.
        """
        assert node is not None
        assert not isinstance(node, str)
        assert isinstance(node, Node), node
        try:
            name = node.__class__.__name__
            if vmethod := getattr(self, "visit_" + name, None):
                res = vmethod(node)
                assert (
                    res is None
                ), f"did you meant to implement replace_{name} instead of visit_{name} ?"
            if method := getattr(self, "replace_" + name, None):
                self._replacements.update([name])
                new_nodes = self._call_method(method, node)

            elif name in [
                "Code",
                "Comment",
                "Directive",
                "Example",
                "Fig",
                "Link",
                "MCode",
                "MComment",
                "MImage",
                "MInlineCode",
                "MInlineMath",
                "MImage",
                "MMath",
                "MText",
                "MThematicBreak",
                "Math",
                "Options",
                "SeeAlsoItem",
                "SubstitutionRef",
                "Unimplemented",
                "ReplaceNode",
            ]:
                return [node]
            else:
                new_children = []
                if not hasattr(node, "children"):
                    raise ValueError(f"{node.__class__} has no children {node}")
                for c in node.children:  # type: ignore
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
                if node.children != new_children:  # type: ignore
                    self._cr += 1
                    # print("Replaced !", node.children, new_children)
                node.children = new_children  # type: ignore
                new_nodes = [node]
            assert isinstance(new_nodes, list)
            return new_nodes
        except Exception as e:
            e.add_note(f"visiting {node=}")
            raise


# misc thoughts:
# we will have multiplet type of directive handlers
# from the simpler to more complex.
# handler that want to parse/handle everything by themsleves,
# other that don't care about domain/role.


Handler = Callable[[str], List[Node]]

DIRECTIVE_MAP: Dict[str, Dict[str, List[Handler]]] = {}
BLOCK_DIRECTIVE_MAP: Dict[str, Dict[str, List[Handler]]] = {}


def directive_handler(domain, role):
    def _inner(func):
        DIRECTIVE_MAP.setdefault(domain, {}).setdefault(role, []).append(func)
        return func

    return _inner


def block_directive_handler(domain, role):
    def _inner(func):
        DIRECTIVE_MAP.setdefault(domain, {}).setdefault(role, []).append(func)
        return func

    return _inner


def _x_any_unimplemented_to_verbatim(domain, role, value):
    # print("To implement", domain, role)
    return [MInlineCode(value)]


for role in ("type", "expr", "member", "macro", "enumerator", "func", "data"):
    directive_handler("c", role)(
        lambda value: _x_any_unimplemented_to_verbatim("c", role, value)
    )
for role in (
    "any",
    "attr",
    "class",
    "command",
    "const",
    "data",
    "keyword",
    "doc",
    "exc",
    "file",
    "func",
    "method",
    "enabled",
    "kbd",
    "meth",
    "mod",
    "obj",
    "program",
    "ref",
    "sub",
    "sup",
    "term",
    "samp",  # networkx, ipython
    "rc",  # matplotlib
):
    directive_handler("py", role)(
        lambda value: _x_any_unimplemented_to_verbatim("py", role, value)
    )


# TODO: make that a plugin/extension/generic to the project.
@directive_handler("py", "ghpull")
def py_ghpull_handler(value):
    return [
        MLink(
            children=[MText(f"#{value}")],
            url=f"https://github.com/ipython/ipython/pull/{value}",
            title="",
        )
    ]


# TODO: make that a plugin/extension/generic to the project.
@directive_handler("py", "ghissue")
def py_ghissue_handler(value):
    return [
        MLink(
            children=[MText(f"#{value}")],
            url=f"https://github.com/ipython/ipython/issue/{value}",
            title="",
        )
    ]


@directive_handler("py", "math")
def py_math_handler(value):
    m = MInlineMath(value)
    return [m]


@directive_handler("py", "pep")
def py_pep_hander(value):
    number = int(value)
    target = f"https://peps.python.org/pep-{number:04d}/"
    return [
        MLink(
            children=[MText(f"Pep {number}")],
            url=target,
            title="",
        )
    ]


_MISSING_DIRECTIVES: List[str] = []
_MISSING_INLINE_DIRECTIVES: List[str] = []


class DirectiveVisiter(TreeReplacer):
    """
    A tree replacer to update directives.

    """

    def __init__(
        self,
        qa: str,
        known_refs: FrozenSet[RefInfo],
        local_refs,
        substitution_defs,
        aliases,
        version,
        config=None,
    ):
        """
        qa: str
            current object fully qualified name
        known_refs: set of RefInfo
            list of all currently know objects
        locals_refs :
            pass
        aliases :
            pass
        version : str
            current version when linking

        """
        assert isinstance(qa, str), qa
        assert isinstance(known_refs, (set, frozenset)), known_refs
        assert isinstance(local_refs, (set, frozenset)), local_refs

        self._handlers = {
            "math": block_math_handler,
            "warning": warning_handler,
            "note": note_handler,
            "versionadded": versionadded_handler,
            "versionchanged": versionchanged_handler,
            "deprecated": deprecated_handler,
        }

        for k, v in config.items():
            self._handlers[k] = obj_from_qualname(v)

        self.known_refs = frozenset(known_refs)
        self.local_refs = frozenset(local_refs)
        self.substitution_defs = substitution_defs
        self.qa = qa
        self.local: List[str] = []
        self.total: List[Tuple[Any, str]] = []
        # long -> short
        self.aliases: Dict[str, str] = aliases
        # short -> long
        self.rev_aliases = {v: k for k, v in aliases.items()}
        self._targets: Set[Any] = set()
        self.version = version
        self._tocs: Any = []

    def replace_Code(self, code):
        """Here we'll return MySt Code."""
        code_ = "".join([entry.value for entry in code.entries])
        return [MCode(code_)]

    def _block_verbatim_helper(self, name: str, argument: str, options: dict, content):
        data = f".. {name}:: {argument}\n"
        for k, v in options.items():
            data = data + f"    :{k}:{v}\n"
        data = data + indent(content, "    ")
        return [MCode(data)]

    def _autosummary_handler(self, argument, options: dict, content):
        # assert False
        return self._block_verbatim_helper("autosummary", argument, options, content)

    def _code_handler(
        self, argument: str, options: Dict[str, str], content: str
    ) -> List[MCode]:
        return [MCode(content)]

    def _toctree_handler(self, argument, options, content):
        assert not argument
        toc = []

        lls = []

        for line in content.splitlines():
            line = line.strip()
            if line == "self":
                # TODO in toctree one line can refer to self
                continue
            if "<" in line and line.endswith(">"):
                title, url = line[:-1].split("<")
                title = title.strip()
                assert "<" not in url
                toc.append([title, url])
                link = Link(
                    title,
                    reference=RefInfo(module="", version="", kind="?", path=url),
                    kind="exists",
                    exists=True,
                    anchor=None,
                )
                RESOLVER.add_reference(link, url)
                lls.append(link)
            else:
                assert "<" not in line
                toc.append([None, line])
                link = Link(
                    line,
                    reference=RefInfo(module="", version="", kind="?", path=line),
                    kind="exists",
                    exists=True,
                    anchor=None,
                )
                RESOLVER.add_reference(link, line)
                lls.append(link)

        self._tocs.append(toc)

        acc = []
        for line in lls:
            acc.append(MListItem(False, [MParagraph([line])]))
        return [MList(ordered=False, start=1, spread=False, children=acc)]

    def replace_UnprocessedDirective(self, directive: UnprocessedDirective):
        meth = getattr(self, "_" + directive.name + "_handler", None)
        if meth is None:
            meth = self._handlers.get(directive.name, None)
        if meth:
            # TODO: we may want to recurse here on returned items.
            res = meth(
                directive.args,
                directive.options,
                directive.value,
            )
            assert isinstance(res, list)
            acc = []
            for a in res:
                # I believe here we may want to wrap things in Paragraph  and comact words ?
                tr = self.generic_visit(a)
                acc.extend(tr)
            return acc

        if directive.name not in _MISSING_DIRECTIVES:
            _MISSING_DIRECTIVES.append(directive.name)
            log.debug("TODO: %s", directive.name)

        return [MMystDirective.from_unprocessed(directive)]

    def replace_MMystDirective(self, myst_directive: MMystDirective):
        """
        Detects MyST directives that are not handled, and adds them to the
        list of missing directives.
        """
        meth = getattr(self, "_" + myst_directive.name + "_handler", None)
        if meth:
            assert False, "should have gone via unprocessed"
        if myst_directive.name not in _MISSING_DIRECTIVES:
            _MISSING_DIRECTIVES.append(myst_directive.name)
            log.debug("TODO: %s", myst_directive.name)

        return [myst_directive]

    def replace_SubstitutionDef(self, node):
        """
        This node should be removed since the actual reference has already been
        resolved.
        """
        return []

    def replace_SubstitutionRef(self, directive):
        # Return the corresponding substitution definition.
        # TODO: This should be either Text (from the replace directive) or
        # an image (from the image directive.)
        if directive.value in self.substitution_defs:
            if isinstance(self.substitution_defs[directive.value][0], ReplaceNode):
                return [MText(self.substitution_defs[directive.value][0].text)]
            elif isinstance(self.substitution_defs[directive.value][0], MImage):
                return [self.substitution_defs[directive.value][0]]
            else:
                raise NotImplementedError(
                    f"SubstitutionDef for custom inline directive {self.substitution_defs[directive.value][0]} not implemented."
                )
        else:
            return [MText(directive.value)]

    def _resolve(self, loc, text):
        """
        Resolve `text` within local references `loc`

        """
        assert isinstance(text, str)
        return resolve_(
            self.qa, self.known_refs, loc, text, rev_aliases=self.rev_aliases
        )

    @classmethod
    def _import_solver(cls, maybe_qa: str):
        parts = maybe_qa.split(".")
        are_id = [x.isidentifier() for x in parts]

        if not all(are_id):
            return None
        else:
            target_qa = full_qual(_obj_from_path(parts))
            if target_qa is not None:
                return target_qa

    def replace_Directive(self, directive: Directive):
        """
        Returns a new node to replace the directive.
        """
        domain, role = directive.domain, directive.role
        if domain is None:
            domain = "py"
        if role is None:
            role = "py"
        domain_handler: Dict[str, List[Handler]] = DIRECTIVE_MAP.get(domain, {})
        handlers: List[Handler] = domain_handler.get(role, [])
        for h in handlers:
            res = h(directive.value)
            if res is not None:
                return res

        loc: FrozenSet[str]
        if directive.role not in ["any", None]:
            loc = frozenset()
        else:
            loc = self.local_refs
        text = directive.value
        assert "`" not in text
        text = text.replace("\n", " ")
        to_resolve = text

        if (
            ("<" in text)
            and text.endswith(">")
            and " <" not in text
            and "\n<" not in text
        ):
            pass  # assert False, ("error space-< in", self.qa, directive)
        if (" <" in text) and text.endswith(">"):
            try:
                text, to_resolve = text.split(" <")
                text = text.rstrip()
            except ValueError:
                assert False, directive.value
            assert to_resolve.endswith(">"), (text, to_resolve)
            to_resolve = to_resolve.rstrip(">")
        elif ("\n <" in text) and text.endswith(">"):
            try:
                text, to_resolve = text.split(" <")
                text = text.rstrip()
            except ValueError:
                assert False, directive.value
            assert to_resolve.endswith(">"), (text, to_resolve)
            to_resolve = to_resolve.rstrip(">")

        if to_resolve.startswith(("https://", "http://", "mailto://")):
            to_resolve = to_resolve.replace(" ", "")
            return [
                MLink(
                    children=[MText(text)],
                    url=to_resolve,
                    title="",
                )
            ]

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
        if (directive.domain, directive.role) in [
            (None, None),
            (None, "mod"),
            (None, "func"),
            (None, "any"),
            (None, "meth"),
            (None, "class"),
        ]:
            text = directive.value
            tqa = directive.value

            if text.startswith("@"):
                tqa = tqa[1:]
            if text.startswith("~"):
                tqa = tqa[1:]
                text = tqa.split(".")[-1]
            # TODO: this may not be correct, is it's start with `.` it should be relative to current object.
            if tqa.startswith("."):
                tqa = tqa[1:]
            if tqa.endswith("()"):
                tqa = tqa[:-2]

            target_qa = self._import_solver(tqa)
            if target_qa is not None:
                if target_qa.split(".")[0] == self.qa.split("."):
                    assert False, "local reference should have explicit versions"
                module = target_qa.split(":")[0].split(".")[0]
                ri = RefInfo(
                    module=module,
                    version="*",
                    kind="api",
                    path=target_qa,
                )
                # print("Solve ri", ri, directive.value, self.qa)
                return [Link(text, ri, "module", True)]
            # print("Not all identifier", directive, "in", self.qa)
        else:
            pass
            # print(
            #    "could not match",
            #    directive,
            #    (directive.role, directive.domain),
            #    "in ",
            #    self.qa,
            # )
        # print("Unchanged:", directive.domain, directive.role, directive.value, self.qa)
        return [directive]


def _import_max(parts):
    p = parts[0]
    try:
        __import__(p)
    except (ImportError, RuntimeError):
        return
    for k in parts[1:]:
        p = p + "." + k
        try:
            __import__(p)
        except (ImportError, RuntimeError):
            return
        except Exception as e:
            raise type(e)(parts)


def _obj_from_path(parts):
    _import_max(parts)
    try:
        target = __import__(parts[0])
        for p in parts[1:]:
            target = getattr(target, p)
    except Exception:
        return
    return target


class DVR(DirectiveVisiter):
    def visit_Section(self, node):
        if node.target:
            # print("Section has target:", node.target)
            # TODO: This is wrong, we should likely change this to the current module that get visited.
            # it likely only affects narative
            RESOLVER.add_target(
                RefInfo("papyri", "0.0.8", "docs", node.target), node.target
            )

    def replace_Fig(self, fig):
        # todo: add version number here
        self._targets.add(fig.value)

        return [fig]


class PostDVR(DirectiveVisiter):
    def replace_Code(self, code):
        assert False

    def replace_Directive(self, d):
        if (d.domain, d.role) not in _MISSING_INLINE_DIRECTIVES:
            _MISSING_INLINE_DIRECTIVES.append((d.domain, d.role))
            log.info("TODO: %r %r %r", d.domain, d.role, d.value)
        return [d]

    def replace_RefInfo(self, refinfo):
        print(refinfo)
        return [refinfo]

    def replace_BlockDirective(self, block_directive: MMystDirective):
        assert False, "should be unreachable"

    def replace_MMystDirective(self, myst_directive: MMystDirective):
        if myst_directive.name not in _MISSING_DIRECTIVES:
            _MISSING_DIRECTIVES.append(myst_directive.name)
            log.info("TODO: %r", myst_directive.name)

        return [myst_directive]
