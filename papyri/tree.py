"""
This module contains utilities to manipulate the documentation units,
usually trees, and update nodes.

"""

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Collection
from functools import lru_cache, partial
from pathlib import Path
from textwrap import indent
from typing import Any, TypeVar, cast

from .directives import (
    admonition_handler,
    attention_handler,
    block_math_handler,
    caution_handler,
    code_handler,
    container_handler,
    csv_table_handler,
    danger_handler,
    deprecated_handler,
    error_handler,
    hint_handler,
    important_handler,
    list_table_handler,
    literalinclude_handler,
    make_figure_handler,
    make_image_handler,
    make_include_handler,
    make_plot_handler,
    note_handler,
    only_handler,
    raw_handler,
    role_unset,
    rubric_handler,
    seealso_handler,
    tip_handler,
    topic_handler,
    versionadded_handler,
    versionchanged_handler,
    warning_handler,
)
from .error_collector import (
    W_MALFORMED_DIRECTIVE,
    W_MISSING_GITHUB_SLUG,
    W_NONSTANDARD_ROLE,
    W_UNKNOWN_ROLE,
    W_UNRESOLVED_DEFAULT_ROLE,
    W_UNRESOLVED_REF,
    W_UNSET_ROLE,
    W_UNSUPPORTED_SUBSTITUTION,
    DiagnosticConfig,
    Diagnostics,
)
from .node_base import Node

_N = TypeVar("_N", bound=Node)
from .nodes import (
    BulletList,
    CitationReference,
    Code,
    CrossRef,
    Directive,
    FootnoteReference,
    InlineCode,
    InlineMath,
    InlineRole,
    Link,
    ListItem,
    LocalRef,
    Paragraph,
    ParamRef,
    RefInfo,
    Section,
    SubstitutionDef,
    SubstitutionRef,
    Text,
    UnprocessedDirective,
)
from .utils import Canonical, FullQual, full_qual, obj_from_qualname

log = logging.getLogger("papyri")


_cache: dict[int, tuple[dict[str, RefInfo], frozenset[str]]] = {}


# @lru_cache(maxsize=100000)
def _build_resolver_cache(
    known_refs: frozenset[RefInfo],
) -> tuple[dict[str, RefInfo], frozenset[str]]:
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

    _map: dict[str, list[RefInfo]] = defaultdict(lambda: [])
    assert isinstance(known_refs, frozenset)
    for k in known_refs:
        assert isinstance(k, RefInfo)
        _map[k.path].append(k)
        # full_qual() uses "module:qualname" colon notation (e.g. "numpy:sin"),
        # but RST inline roles produce dot notation (e.g. ":func:`numpy.sin`"
        # → "numpy.sin").  Index the dot form as an alias so that both notations
        # resolve to the same RefInfo.
        if ":" in k.path:
            _map[k.path.replace(":", ".")].append(k)

    _m2: dict[str, RefInfo] = {}
    for kk, v in _map.items():
        cand = list(sorted(v, key=lambda x: "" if x.version is None else x.version))
        assert len({c.module for c in cand}) == 1, cand
        _m2[kk] = cand[-1]

    return _m2, frozenset(_m2.keys())


@lru_cache(10000)
def root_start(root: str, refs: frozenset[str]) -> frozenset[str]:
    """
    Compute a subset of references that start with given root.
    """
    return frozenset(r for r in refs if r.startswith(root))


@lru_cache(10000)
def endswith(end: str, refs: frozenset[str]) -> frozenset[str]:
    """
    Compute as subset of references that ends with given root.
    """
    return frozenset(r for r in refs if r.endswith(end))


class DelayedResolver:
    _targets: dict[str, RefInfo | LocalRef]
    _references: dict[str, list[CrossRef]]

    def __init__(self) -> None:
        self._targets = dict()
        self._references = dict()

    def add_target(self, target_ref: RefInfo | LocalRef, target: str) -> None:
        assert target is not None
        assert target not in self._targets, "two targets with the same name"
        self._targets[target] = target_ref
        self._resolve(target)

    def add_reference(self, link: CrossRef, target: str) -> None:
        self._references.setdefault(target, []).append(link)
        self._resolve(target)

    def _resolve(self, target: str) -> None:
        if (target in self._targets) and (target in self._references):
            for link in self._references[target]:
                link.reference = self._targets[target]
            self._references[target] = []


RESOLVER = DelayedResolver()


def resolve_(
    qa: str,
    known_refs: frozenset[RefInfo],
    local_refs: frozenset[str],
    ref: str,
    rev_aliases: dict[Canonical, FullQual],
) -> RefInfo:
    """
    Given the current context (qa), and a str (ref), compute the RefInfo object.

    References are often relative based on the current context (which object you
    are currently in).

    Given this information and all the local (same document) and global
    (same library/all libraries) references, compute the Reference Info object.

    Parameters
    ----------
    qa : str
        fully qualified path of the current object (.value).
        TODO: this will be weird for non object, like example.
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
    hk = hash(known_refs)
    hash(local_refs)
    assert rev_aliases is not None
    ref = Canonical(ref)
    if ref in rev_aliases:
        new_ref = rev_aliases[ref]
        # rev_aliases is keyed by Canonical; the alias target is a
        # FullQual. A direct ``new_ref not in rev_aliases`` compared
        # FullQual against Canonical keys, so the cycle guard never
        # fired. The recursive call below is already safe (empty
        # rev_aliases), so cycles cannot actually occur here — drop
        # the defensive assert rather than keep a silent no-op.
        res = resolve_(qa, known_refs, local_refs, new_ref, {})
        return res

    assert isinstance(ref, str), ref

    # TODO: LRU Cache seem to have speed problem here; and get slow while this should be just fine.
    # this seem to be due to the fact that even if the hash is the same this still needs to compare the objects, as
    # those may have been muted.
    if hk not in _cache:
        _cache[hk] = _build_resolver_cache(known_refs)

    # qa uses full_qual "module:qualname" notation ("numpy:any",
    # "numpy.ma.core:MaskedArray.var") while the lookup keys are indexed in
    # dotted form; normalize before deriving enclosing scopes, otherwise no
    # relative ref inside an object page can ever resolve.
    qa = qa.replace(":", ".")

    # this is a mapping from the key to the most relevant
    # Refinfo to a document
    k_path_map: dict[str, RefInfo]

    # hashable for caching /optimisation.
    keyset: frozenset[str]

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
            # ~.Foo.Bar.Baz style: dot is a "start from root" hint, the
            # remainder is an absolute path.
            if (abs_ref := ref[1:]) in k_path_map:
                return k_path_map[abs_ref]
            else:
                root = qa.split(".")[0]
                sub1 = root_start(root, keyset)
                # Suffix search on a *component boundary*: ".mod.Name" so
                # that "pkg.mod.Name" matches but "pkg.altmod.Name" via a
                # bare-string suffix ("od.Name") cannot. A bare endswith
                # let ".shape" match "numpy.reshape" — a silently wrong
                # link, indistinguishable from an exact hit.
                # Dedupe through the RefInfo: the colon key and its dotted
                # alias both match the suffix but name the same object.
                subset = {k_path_map[q] for q in endswith("." + abs_ref, sub1)}
                if len(subset) == 1:
                    return next(iter(subset))
                # Zero or many hits: unresolved. Ambiguity must surface as
                # a diagnostic at the caller, never an arbitrary pick.
                return RefInfo(None, None, "missing", ref)

        # Walk the enclosing scopes most-specific-first (Sphinx resolves
        # relative to the closest enclosing module/class): try "qa.ref"
        # first, then each parent scope, down to "root.ref". The previous
        # form of this loop computed the attempt *before* extending the
        # prefix, so the current scope itself ("qa.ref") was never tried —
        # a module docstring could not resolve a ref relative to its own
        # module.
        parts = qa.split(".")
        for i in range(len(parts), 0, -1):
            attempt = ".".join(parts[:i]) + "." + ref
            if attempt in k_path_map:
                return k_path_map[attempt]

    # Last chance: an exact component-boundary suffix match anywhere under
    # the bundle root ("linspace" → "numpy.linspace", "Chebyshev.fit" →
    # "numpy.polynomial.chebyshev.Chebyshev.fit"), mirroring Sphinx's
    # suffix search. The historical substring fallback (`ref in q`) is
    # gone: a single accidental substring hit ("cos" inside "numpy.arccos")
    # shipped a silently wrong link, indistinguishable from an exact match
    # — no fuzzy matching may decide what reaches the IR. Ambiguity (two
    # boundary matches) is also unresolved, never an arbitrary pick; the
    # caller emits the diagnostic.
    q0 = parts[0]
    rs = root_start(q0, keyset)
    # Dedupe through the RefInfo: the colon key and its dotted alias both
    # match the suffix but name the same object.
    trail = {k_path_map[q] for q in rs if q.endswith("." + ref)}
    if len(trail) == 1:
        return next(iter(trail))

    return RefInfo(None, None, "missing", ref)


class TreeVisitor:
    def __init__(self, find: Collection[type[Node]]) -> None:
        self.skipped: set[type[Node]] = set()
        self.find = find

    def generic_visit(self, node: Node) -> dict[type[Node], list[Node]]:
        from .nodes import Options, ThematicBreak

        name = node.__class__.__name__
        if method := getattr(self, "visit_" + name, None):
            return cast(dict[type[Node], list[Node]], method(node))
        elif hasattr(node, "children"):
            acc: dict[type[Node], list[Node]] = {}
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
        elif isinstance(
            node,
            (
                LocalRef,
                RefInfo,
                Options,
                ThematicBreak,
                SubstitutionDef,
                CitationReference,
                FootnoteReference,
            ),
        ):
            return {}
        else:
            raise ValueError(f"{node.__class__} has no children, no values {node}")


class TreeReplacer:
    """
    Tree visitor with methods to replace nodes.

    define replace_XXX(xxx) that return a list of new nodes, and call visit(and the root tree)
    """

    _replacements: Counter[str]

    def __init__(self) -> None:
        self._replacements = Counter()

    def visit(self, node: _N) -> _N:
        self._replacements = Counter()
        self._cr = 0
        assert not isinstance(node, list), node
        assert node is not None
        res = self.generic_visit(node)
        assert len(res) == 1, res
        return cast(_N, res[0])

    def _call_method(
        self, method: Callable[[Node], list[Node]], node: Node
    ) -> list[Node]:
        return method(node)

    def generic_visit(self, node: Node) -> list[Node]:
        assert node is not None
        assert not isinstance(node, str)
        assert isinstance(node, Node), node
        try:
            name = node.__class__.__name__
            if vmethod := getattr(self, "visit_" + name, None):
                res = vmethod(node)
                assert res is None, (
                    f"did you meant to implement replace_{name} instead of visit_{name} ?"
                )
            if method := getattr(self, "replace_" + name, None):
                self._replacements.update([name])
                new_nodes = self._call_method(method, node)
            elif name in [
                "Code",
                "Comment",
                "Example",
                "Figure",
                "GenCode",
                "Image",
                "InlineCode",
                "InlineMath",
                "InlineRole",
                "Math",
                "ParamRef",
                "Options",
                "SeeAlsoItem",
                "SubstitutionRef",
                "Target",
                "Text",
                "ThematicBreak",
                "Unimplemented",
                "DocstringSentinel",
                "CrossRef",
                "CitationReference",
                "FootnoteReference",
            ]:
                return [node]
            else:
                new_children = []
                if not hasattr(node, "children"):
                    raise ValueError(f"{node.__class__} has no children {node}")
                # `Node` itself doesn't declare `children`; only subclasses
                # do. The hasattr guard above narrows at runtime; cast
                # tells mypy to accept the attribute without a brittle
                # type-ignore (which `warn_unused_ignores` may flag on
                # Python 3.14).
                node_with_children = cast(Any, node)
                children: list[Node] = node_with_children.children
                for c in children:
                    assert c is not None, f"{node=} has a None child"
                    assert isinstance(c, Node), c
                    replacement = self.generic_visit(c)
                    assert isinstance(replacement, list)

                    new_children.extend(replacement)
                if tuple(children) != tuple(new_children) and hasattr(self, "_cr"):
                    self._cr += 1
                node_with_children.children = tuple(new_children)
                # ``Section.title`` holds inline nodes outside ``children``, so
                # the recursion above never reaches them; without this a
                # SubstitutionRef or role in a heading escapes every
                # replacement pass and lands in the IR verbatim.
                if title := getattr(node, "title", None):
                    new_title: list[Node] = []
                    for t in title:
                        new_title.extend(self.generic_visit(t))
                    node_with_children.title = tuple(new_title)
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


Handler = Callable[[str], list[Node]]

DIRECTIVE_MAP: dict[str, dict[str, list[Handler]]] = {}


def directive_handler(domain: str, role: str) -> Callable[[Any], Any]:
    def _inner(func: Any) -> Any:
        DIRECTIVE_MAP.setdefault(domain, {}).setdefault(role, []).append(func)
        return func

    return _inner


def _x_any_unimplemented_to_verbatim(domain: str, role: str, value: str) -> list[Any]:
    return [InlineCode(value)]


# C-domain roles: we don't index C symbols, so resolve can never succeed —
# emit verbatim InlineCode directly.
for role in ("type", "expr", "member", "macro", "enumerator", "func", "data"):
    directive_handler("c", role)(
        lambda value, _role=role: _x_any_unimplemented_to_verbatim("c", _role, value)
    )

# Formatting-only roles: these are not cross-references, they just affect
# visual rendering (sub/superscript, keyboard keys, filenames, literals, ...).
# Emit verbatim InlineCode so they never enter the resolve path — it would
# always fail and pollute the "missing" diagnostics.
_PY_VERBATIM_ROLES = (
    "command",
    "enabled",
    "file",
    "kbd",
    "keyword",
    "program",
    "rc",  # matplotlib
    "samp",  # networkx, ipython
    "sub",
    "sup",
    "term",
    # Standard Sphinx std-domain / docutils formatting roles. They are part
    # of the documented RST/Sphinx vocabulary (not project inventions), so
    # they are built in rather than requiring every project to map them in
    # [global.roles]; papyri has no target index for them, so render as code.
    "abbr",
    "code",  # docutils inline code role
    "dfn",
    "envvar",
    "guilabel",
    "mailheader",
    "makevar",
    "manpage",
    "menuselection",
    "mimetype",
    "newsgroup",
    "option",
    "regexp",
    "token",
    # Sphinx math-domain references (":eq:`label`", ":math:numref:`fig`").
    # papyri keeps no equation/figure-number index, so the label renders as
    # code rather than pretending to link.
    "eq",
    "numref",
)

# Cross-reference roles (any/attr/class/const/data/exc/func/meth/method/mod/obj
# and the C-domain equivalents) are handled by the resolve path in
# ``DirectiveVisiter.replace_InlineRole``; registering a verbatim handler here
# would short-circuit that path and prevent crosslinks from ever being
# generated.  ``ref`` (section-label refs) also falls through: if resolve can't
# find the target the original ``InlineRole`` is returned and rendered as
# styled code, matching the previous verbatim appearance.
# Registered under "py" (the default when a role is written without a
# domain), "std" (their real Sphinx domain, for the explicit
# ``:std:envvar:`` spelling) and "math" (``:math:numref:`` / ``:math:eq:``).
for domain in ("py", "std", "math"):
    for role in _PY_VERBATIM_ROLES:
        directive_handler(domain, role)(
            lambda value, _domain=domain, _role=role: _x_any_unimplemented_to_verbatim(
                _domain, _role, value
            )
        )


# :ghpull: / :ghissue: are IPython-invented roles honoured for any project
# whose config declares ``[meta].github_slug``. They are resolved in
# ``DirectiveVisiter.replace_InlineRole`` (not registered in the global
# ``DIRECTIVE_MAP``) because they need per-bundle state — the slug and the
# ``Diagnostics`` collector — rather than reading module-level globals. When no
# slug is configured the role renders as plain ``#N`` text and emits a
# ``W-missing-github-slug`` diagnostic so maintainers notice the missing config
# instead of getting a silent link to the wrong repo (historically hardcoded to
# ``ipython/ipython``, which was correct for exactly one project).
_GH_ROLE_PATH_SEGMENT = {"ghpull": "pull", "ghissue": "issues"}

# Roles that name a Python object and are resolved by the cross-reference
# machinery in ``replace_InlineRole`` (in-bundle lookup, then the import
# solver). The domain must be None (role written without a domain prefix,
# e.g. :func:`…`) or "py" (explicit Python domain, e.g. :py:func:`…`).
# ``None`` is the bare default role.
_PYTHON_OBJECT_ROLES = frozenset(
    {
        None,
        "mod",
        "func",
        "any",
        "meth",
        # papyri-accepted long form of :meth:, seen in docstrings in the
        # wild even though Sphinx rejects it. Resolves like :meth: but every
        # use emits W-nonstandard-role (see replace_InlineRole).
        "method",
        "class",
        "exc",
        "data",
        "attr",
        "obj",
        "const",
    }
)


@directive_handler("py", "math")
def py_math_handler(value: str) -> list[Any]:
    m = InlineMath(value)
    return [m]


@directive_handler("py", "pep")
def py_pep_hander(value: str) -> list[Any]:
    number = int(value)
    target = f"https://peps.python.org/pep-{number:04d}/"
    return [
        Link(
            children=[Text(f"Pep {number}")],
            url=target,
            title="",
        )
    ]


@directive_handler("py", "param")
def py_param_handler(value: str) -> list[Any]:
    """Handle ``:param:`name``` — an inline reference to a sibling parameter."""
    return [ParamRef(name=value.strip())]


_MISSING_DIRECTIVES: list[str] = []

_SPHINX_ONLY_DIRECTIVES: frozenset[str] = frozenset(
    {
        # autodoc directives — not meaningful outside a Sphinx build
        "autofunction",
        "autoclass",
        "autoattribute",
        "autodata",
        "autoexception",
        "automodule",
        # doctest infrastructure — drop; content is not documentation prose
        "testsetup",
        "testcleanup",
        "testcode",
        "testoutput",
        # presentation hint — safe to drop (highlight language is render-side)
        "highlight",
        # currentmodule shifts cross-ref resolution; drop to avoid raw Directive nodes
        # until full ref-resolution support is added (see PLAN.md)
        "currentmodule",
        # Sphinx py-domain manual API directives (handwritten, no "auto" prefix)
        "py:function",
        "py:class",
        "py:method",
        "py:attribute",
        "py:data",
        "py:exception",
        "py:module",
        # Also handle without explicit domain prefix (common in older scipy/numpy docs)
        "function",
        "class",
        "method",
        "attribute",
        "data",
        "exception",
        "module",
        # sphinx-design — pure HTML-layout directives used on PyData-theme landing
        # pages (e.g. numpy / scipy `doc/source/index.rst`). They have no IR
        # equivalent and their body is decorative wrapping around links the
        # toctree already provides; drop the whole block so a single grid in
        # the root index doesn't take down the page (and with it the toc).
        "grid",
        "grid-item",
        "grid-item-card",
        "card",
        "card-carousel",
        "tab-set",
        "tab-item",
        "dropdown",
        "button-link",
        "button-ref",
    }
)


class DirectiveVisiter(TreeReplacer):
    """
    A tree replacer to update directives.

    """

    def __init__(
        self,
        qa: str,
        known_refs: frozenset[RefInfo],
        local_refs: frozenset[str] | set[str],
        aliases: dict[str, str],
        version: str,
        config: dict[str, str | dict[str, Any]] | None = None,
        module: str | None = None,
        doc_path: Path | None = None,
        asset_store: Callable[[str, bytes], None] | None = None,
        doc_root: Path | None = None,
        doc_targets: dict[str, str] | None = None,
        external_targets: dict[str, str] | None = None,
        doc_titles: dict[str, str] | None = None,
        execute: bool = False,
        param_names: frozenset[str] | set[str] | None = None,
        diagnostics: Diagnostics | None = None,
        github_slug: str | None = None,
        roles: dict[str, str] | None = None,
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
        module : str, optional
            root module name being documented; derived from qa when omitted

        """
        assert isinstance(qa, str), qa
        assert isinstance(known_refs, (set, frozenset)), known_refs
        assert isinstance(local_refs, (set, frozenset)), local_refs

        self._handlers: dict[str, Callable[..., Any]] = {
            "math": block_math_handler,
            "warning": warning_handler,
            "note": note_handler,
            "seealso": seealso_handler,
            "versionadded": versionadded_handler,
            "versionchanged": versionchanged_handler,
            "deprecated": deprecated_handler,
            "code-block": code_handler,
            # Docutils alias of ``code-block``; common in IPython/Sphinx docs
            # as ``.. sourcecode:: ipython``.
            "sourcecode": code_handler,
            "code": code_handler,
            "list-table": partial(list_table_handler, warn=self._directive_warn),
            "rubric": rubric_handler,
            "only": only_handler,
            "literalinclude": literalinclude_handler,
            "csv-table": partial(
                csv_table_handler,
                doc_path=doc_path,
                doc_root=doc_root,
                warn=self._directive_warn,
            ),
            # Standard RST admonitions not yet handled above.
            "attention": attention_handler,
            "caution": caution_handler,
            "danger": danger_handler,
            "error": error_handler,
            "hint": hint_handler,
            "important": important_handler,
            "tip": tip_handler,
            # Generic admonition with explicit title.
            "admonition": admonition_handler,
            # Self-contained mini-section.
            "topic": topic_handler,
            # Raw output-format content — always drop (security risk).
            "raw": raw_handler,
            # Structural wrapper — drop the container, keep the children.
            "container": container_handler,
        }

        # Handlers that close over ``self`` — registered here so the dict is
        # the sole dispatch mechanism (no getattr fallback needed).
        self._handlers["toctree"] = self._toctree_handler
        self._handlers["autosummary"] = self._autosummary_handler

        for k, v in (config or {}).items():
            if isinstance(v, str):
                self._handlers[k] = obj_from_qualname(v)
            else:
                handler_ref = v["handler"]
                ctor_args = tuple(v.get("init_args") or ())
                ctor_kwargs: dict[str, Any] = v.get("init_kwargs") or {}
                self._handlers[k] = obj_from_qualname(
                    handler_ref, ctor_args, ctor_kwargs
                )

        # ``[global.roles]`` — project-local inline roles ("mpltype" or
        # "domain:role") mapped to a handler that receives the role body.
        # Consulted in ``replace_InlineRole`` before the built-in registry.
        self._role_handlers: dict[str, Callable[[str], list[Any] | None]] = {}
        for k, v in (roles or {}).items():
            handler = obj_from_qualname(v)
            if handler is role_unset:
                # The declared placeholder: bind the role name and route its
                # warning through Diagnostics as W-unset-role so every use
                # stays visible. ``_unset_role_warn`` reads ``self.diagnostics``
                # / ``self.qa`` at call time (both are assigned below).
                handler = partial(role_unset, role=k, warn=self._unset_role_warn)
            self._role_handlers[k] = handler

        self.known_refs = frozenset(known_refs)
        self.local_refs = frozenset(local_refs)
        # Coded gen-time diagnostics. Defaults to a standalone collector (every
        # code at its registered severity) so the visitor still works when
        # driven outside ``papyri gen`` — e.g. tests and ``papyri.tests.utils``.
        self.diagnostics: Diagnostics = (
            diagnostics
            if diagnostics is not None
            else Diagnostics(DiagnosticConfig.default(), log)
        )
        # ``owner/name`` GitHub slug from ``[meta].github_slug``, used by the
        # ``:ghpull:`` / ``:ghissue:`` roles in ``replace_InlineRole``. ``None``
        # means the roles render as plain ``#N`` text and emit a diagnostic.
        self.github_slug: str | None = github_slug or None
        self.qa = qa
        # qa may use either `.` (submodule path) or `:` (top-level module
        # attribute, e.g. "numpy:promote_types") as the first separator;
        # both must be split to extract the bundle's root module name.
        self.module: str = (
            module if module is not None else qa.split(".")[0].split(":")[0]
        )
        self.local: list[str] = []
        self.total: list[tuple[Any, str]] = []
        # long -> short
        self.aliases: dict[str, str] = aliases
        # short -> long
        self.rev_aliases = {v: k for k, v in aliases.items()}
        self._targets: set[Any] = set()
        self.version = version
        self._tocs: Any = []
        # Maps RST target label -> doc key for :ref: resolution within the bundle.
        self.doc_targets: dict[str, str] = (
            doc_targets if doc_targets is not None else {}
        )
        # Maps RST target label -> external URL for named-hyperlink references
        # of the form ``.. _label: http://...`` referenced via ``label_``.
        self.external_targets: dict[str, str] = (
            external_targets if external_targets is not None else {}
        )
        # Maps doc key (':' separated) -> first section title, populated by
        # gen's first parse pass. Toctree entries without an explicit title
        # resolve their display text against this map so the rendered bullet
        # shows the document's heading rather than the raw path.
        self.doc_titles: dict[str, str] = doc_titles if doc_titles is not None else {}
        # Names of parameters in the enclosing callable's signature, used to
        # auto-promote bare backtick references like `url` into ParamRef nodes
        # so the viewer can cross-highlight prose ↔ signature.
        self.param_names: frozenset[str] = (
            frozenset(param_names) if param_names is not None else frozenset()
        )
        # Keyed by RST name with pipes (e.g. '|foo|').  Populated by
        # collect_substitutions() before visiting; can be pre-seeded with
        # config-level global substitutions.
        self._substitutions: dict[str, list[Any]] = {}
        # Register the default ``.. image::`` handler unless the caller already
        # provided one via *config*.  Config-supplied handlers are applied above
        # and win over this default.
        self._handlers.setdefault(
            "image",
            make_image_handler(
                doc_path,
                asset_store,
                self.module,
                self.version,
                doc_root,
                warn=self._directive_warn,
            ),
        )
        self._handlers.setdefault(
            "figure",
            make_figure_handler(
                doc_path,
                asset_store,
                self.module,
                self.version,
                doc_root,
                warn=self._directive_warn,
            ),
        )
        self._handlers.setdefault(
            "include",
            make_include_handler(doc_path, doc_root, warn=self._directive_warn),
        )
        self._handlers.setdefault(
            "plot",
            make_plot_handler(
                asset_store=asset_store,
                module=self.module,
                version=self.version,
                execute=execute,
                qa=self.qa,
                doc_path=doc_path,
                doc_root=doc_root,
                warn=self._directive_warn,
            ),
        )

    def _gh_link_or_warn(self, role: str, value: str) -> list[Any]:
        """Resolve a ``:ghpull:`` / ``:ghissue:`` role against ``github_slug``.

        Renders a GitHub link when the bundle configured ``[meta].github_slug``;
        otherwise falls back to plain ``#N`` text and emits a
        ``W-missing-github-slug`` diagnostic against the current object.
        """
        if self.github_slug is None:
            self.diagnostics.emit(
                W_MISSING_GITHUB_SLUG,
                self.qa,
                f":{role}: used but [meta].github_slug is not set; rendering "
                f"#{value} as plain text. Add github_slug = 'owner/name' under "
                "[meta] in your config to enable these links.",
            )
            return [Text(f"#{value}")]
        path_segment = _GH_ROLE_PATH_SEGMENT[role]
        return [
            Link(
                children=[Text(f"#{value}")],
                url=f"https://github.com/{self.github_slug}/{path_segment}/{value}",
                title="",
            )
        ]

    def _directive_warn(self, message: str) -> None:
        """Report a malformed/unprocessable directive as a coded diagnostic.

        Bound into the free-function directive handlers (``list-table``,
        ``csv-table``, ``image``, ``figure``, ``include``, ``plot``) so their
        recoverable failures flow through ``Diagnostics`` as
        ``W-malformed-directive`` against the current object, instead of a bare
        ``log.warning``.
        """
        self.diagnostics.emit(W_MALFORMED_DIRECTIVE, self.qa, message)

    def _unset_role_warn(self, message: str) -> None:
        """Report a use of a ``role_unset`` placeholder mapping as ``W-unset-role``.

        Bound as the ``warn`` callback of every ``[global.roles]`` entry that
        maps to ``papyri.directives.role_unset``, so each use is recorded
        against the current object rather than logged loosely.
        """
        self.diagnostics.emit(W_UNSET_ROLE, self.qa, message)

    def collect_substitutions(self, *sections: Section) -> None:
        """Pre-scan sections for SubstitutionDef nodes to build the substitution map.

        Call this on all sections that will be visited *before* calling visit(),
        so that refs are resolved even when the def appears after the ref in
        document order.
        """
        for section in sections:
            for node in section.children:
                if not isinstance(node, SubstitutionDef):
                    continue
                child = node.children[0] if node.children else None
                if isinstance(child, UnprocessedDirective) and child.name == "replace":
                    replacement_text = child.args or ""
                    self._substitutions[node.value] = (
                        [Text(replacement_text)] if replacement_text else []
                    )
                else:
                    directive_name = (
                        child.name
                        if isinstance(child, UnprocessedDirective)
                        else type(child).__name__
                    )
                    self.diagnostics.emit(
                        W_UNSUPPORTED_SUBSTITUTION,
                        self.qa,
                        f"substitution {node.value!r} uses unsupported directive "
                        f"{directive_name!r}; dropping",
                    )

    def replace_SubstitutionDef(self, node: SubstitutionDef) -> list[Any]:
        return []

    def replace_Comment(self, node: Any) -> list[Any]:
        # RST ``.. comment text`` is editor-side prose with no rendered output.
        # Dropping it at visit time keeps Comment nodes out of containers that
        # would otherwise have to allow it everywhere, and out of the IR shape
        # that the viewer has to know about.
        return []

    def replace_SubstitutionRef(self, node: SubstitutionRef) -> list[Any]:
        name = node.value  # e.g. '|foo|'
        if name in self._substitutions:
            return list(self._substitutions[name])
        self.diagnostics.emit(
            W_UNRESOLVED_REF,
            self.qa,
            f"unresolved substitution reference {name!r}",
        )
        inner = name[1:-1] if name.startswith("|") and name.endswith("|") else name
        return [Text(inner)]

    def replace_GenCode(self, code: Any) -> list[Any]:
        """Flatten a GenCode intermediate into a plain Code node."""
        code_ = "".join([entry.value for entry in code.entries])
        status = (
            code.ce_status.value if hasattr(code.ce_status, "value") else code.ce_status
        )
        return [Code(code_, status, code.out)]

    def _block_verbatim_helper(
        self, name: str, argument: str, options: dict[str, str], content: str
    ) -> list[Code]:
        data = f".. {name}:: {argument}\n"
        for k, v in options.items():
            data = data + f"    :{k}:{v}\n"
        data = data + indent(content, "    ")
        return [Code(data)]

    def _autosummary_handler(
        self, argument: str, options: dict[str, str], content: str
    ) -> list[Code]:
        # assert False
        return self._block_verbatim_helper("autosummary", argument, options, content)

    def _resolve_doc_path(self, path: str) -> str:
        """Resolve a toctree entry to a doc key (':' separator).

        Toctree entries are paths relative to the current document's
        directory; a leading '/' anchors to the source root. The doc key
        joins directory parts with ':' (see ``collect_narrative_docs``).
        """
        if path.endswith(".rst"):
            path = path[:-4]
        if path.startswith("/"):
            parts = [p for p in path.lstrip("/").split("/") if p]
        else:
            parent_parts = self.qa.split(":")[:-1] if ":" in self.qa else []
            parts = parent_parts + [p for p in path.split("/") if p]
        return ":".join(parts)

    def _toctree_crossref(self, text: str, path: str) -> CrossRef:
        return CrossRef(
            text,
            reference=LocalRef("docs", self._resolve_doc_path(path)),
            kind="exists",
        )

    def _toctree_handler(
        self, argument: str | None, options: Any, content: str
    ) -> list[Any]:
        # argument is ignored (rare cases like ``.. toctree:: My Title``).
        toc: list[list[str | None]] = []
        lls = []

        opts = options if isinstance(options, dict) else {}
        glob = opts.get("glob", False)
        hidden = opts.get("hidden", False)

        for line in content.splitlines():
            line = line.strip()
            # Skip blank lines, comments, and the special "self" entry.
            if not line or line.startswith("..") or line == "self":
                continue
            # Skip glob patterns — we don't expand them at gen time.
            if glob and ("*" in line or "?" in line):
                continue

            if "<" in line and line.endswith(">"):
                # "Title Text <path>" form — split on last " <".
                try:
                    title, url = line[:-1].rsplit(" <", 1)
                    title = title.strip()
                except ValueError:
                    continue
                toc.append([title, url])
                lls.append(self._toctree_crossref(title, url))
            elif "<" not in line:
                # No explicit title — show the target document's heading
                # instead of the raw reference path. Falls back to the path
                # when the title is unknown (e.g. forward reference, or doc
                # without a top-level title).
                resolved = self._resolve_doc_path(line)
                display = self.doc_titles.get(resolved, line)
                toc.append([None, line])
                lls.append(self._toctree_crossref(display, line))
            # Lines with "<" but not ending ">" are malformed — skip with a warning.
            else:
                log.warning("toctree: skipping malformed entry %r", line)

        self._tocs.append(toc)

        # hidden toctrees contribute to navigation metadata but are not
        # rendered inline on the page.
        if hidden:
            return []

        acc = [ListItem([Paragraph([line])]) for line in lls]
        # Every entry was filtered out (empty/blank content, only comments or
        # ``self``, or a ``:glob:`` toctree of pure wildcards). Emit nothing
        # rather than an empty ``<ul>`` — it renders invisibly but litters the
        # IR with empty BulletList nodes. The toc metadata is already recorded
        # above, so navigation is unaffected (same contract as ``hidden``).
        if not acc:
            return []
        return [BulletList(ordered=False, start=1, children=acc)]

    def replace_UnprocessedDirective(
        self, directive: UnprocessedDirective
    ) -> list[Any]:
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

        if directive.name in _SPHINX_ONLY_DIRECTIVES:
            log.info(
                "skipping Sphinx-only directive %r in %s (not meaningful outside a Sphinx build)",
                directive.name,
                self.qa,
            )
            return []

        if directive.name not in _MISSING_DIRECTIVES:
            _MISSING_DIRECTIVES.append(directive.name)
            log.debug("TODO: %s", directive.name)

        return [Directive.from_unprocessed(directive)]

    def _resolve(self, loc: frozenset[str], text: str) -> RefInfo:
        """
        Resolve `text` within local references `loc`

        """
        assert isinstance(text, str)
        # Narrative doc keys ("reference:ufuncs") are not Python paths under
        # the package root; resolve those relative to the root module so the
        # in-bundle scope walk and suffix search still apply.
        qa = self.qa
        if not (
            qa == self.module
            or qa.startswith(self.module + ".")
            or qa.startswith(self.module + ":")
        ):
            qa = self.module
        return resolve_(
            qa,
            self.known_refs,
            loc,
            text,
            rev_aliases=cast(dict[Canonical, FullQual], self.rev_aliases),
        )

    def _ref_to_crossref(self, text: str, r: RefInfo, exists: str) -> CrossRef:
        """Convert a resolved RefInfo to a CrossRef node.

        Subclasses may override to substitute LocalRef for same-bundle targets.
        """
        return CrossRef(text, r, exists)

    @classmethod
    def _import_solver(cls, maybe_qa: str) -> str | None:
        parts = maybe_qa.split(".")
        are_id = [x.isidentifier() for x in parts]

        if not all(are_id):
            return None
        else:
            target = _obj_from_path(parts)
            target_qa = full_qual(target)
            if target_qa is not None:
                return target_qa
            if target is not None:
                # Objects with no usable __module__/__qualname__ (e.g. method
                # descriptors like numpy.ufunc.reduce): the successful
                # attribute traversal itself proves the path exists, so derive
                # module:qualname from the longest imported module prefix.
                import sys

                for i in range(len(parts), 0, -1):
                    mod = ".".join(parts[:i])
                    if mod in sys.modules:
                        qual = ".".join(parts[i:])
                        return FullQual(f"{mod}:{qual}") if qual else FullQual(mod)

        # Builtin fallback: names like True, False, None, repr, KeyError, dict
        # that belong to the `builtins` module.  full_qual() returns None for
        # singletons (True/False/None) so we special-case those.  We emit the
        # "builtins:<name>" path; resolveExternalRefs in graph.ts strips the
        # "builtins." prefix to match Python's objects.inv, which registers
        # bare names (repr, not builtins.repr).
        if len(parts) == 1:
            import builtins as _builtins

            if hasattr(_builtins, parts[0]):
                obj = getattr(_builtins, parts[0])
                fq = full_qual(obj)
                return fq if fq is not None else FullQual(f"builtins:{parts[0]}")
        return None

    def replace_InlineRole(self, directive: InlineRole) -> list[Any]:
        # Bare interpreted text (no domain, no role) whose value names a
        # parameter of the enclosing callable is promoted to a ParamRef so
        # the viewer can cross-highlight prose ↔ signature.
        if (
            directive.domain is None
            and directive.role is None
            and directive.value in self.param_names
        ):
            return [ParamRef(name=directive.value)]
        domain, role = directive.domain, directive.role
        if domain is None:
            domain = "py"
        if role is None:
            role = "py"
        if domain == "py" and role in _GH_ROLE_PATH_SEGMENT:
            return self._gh_link_or_warn(role, directive.value)
        # Config-supplied role handlers ([global.roles]) win over the
        # built-in registry; keyed by bare role name or "domain:role".
        if self._role_handlers and directive.role is not None:
            key = (
                f"{directive.domain}:{directive.role}"
                if directive.domain
                else directive.role
            )
            if (rh := self._role_handlers.get(key)) is not None:
                res = rh(directive.value)
                if res is not None:
                    return res
        domain_handler: dict[str, list[Handler]] = DIRECTIVE_MAP.get(domain, {})
        handlers: list[Handler] = domain_handler.get(role, [])
        for h in handlers:
            res = h(directive.value)
            if res is not None:
                return res

        # Any role still unhandled here is *unknown*: not a built-in, not
        # config-mapped, and not one of the cross-reference roles resolved
        # below (default role, py object roles, :ref:, :doc:). Do not fall
        # through to resolution — an unknown role accidentally matching a
        # known path would silently cross-link, the implicit behaviour that
        # plagues Sphinx. Every role must be an explicit decision: register
        # it in [global.roles] or downgrade W-unknown-role (error by
        # default, so gen fails fast).
        is_crossref_role = directive.role in ("ref", "doc") or (
            directive.role in _PYTHON_OBJECT_ROLES and directive.domain in (None, "py")
        )
        if not is_crossref_role:
            role_key = (
                f"{directive.domain}:{directive.role}"
                if directive.domain
                else directive.role
            )
            self.diagnostics.emit(
                W_UNKNOWN_ROLE,
                self.qa,
                f"unknown role :{role_key}: — no handler registered; map it in "
                f"[global.roles] (papyri.directives:role_verbatim / role_text / "
                f"role_drop or a custom handler)",
            )
            return [directive]

        # Accepted-but-nonstandard spellings resolve normally but never
        # silently: Sphinx would reject them, so the docstring is wrong.
        if directive.role == "method":
            self.diagnostics.emit(
                W_NONSTANDARD_ROLE,
                self.qa,
                f"role :method: is not a Sphinx role — use :meth: "
                f"({directive.value!r})",
            )

        loc: frozenset[str]
        loc = frozenset() if directive.role not in ["any", None] else self.local_refs
        text = directive.value
        assert "`" not in text
        text = text.replace("\n", " ")
        to_resolve = text

        # Sphinx: a leading "!" on the raw role text suppresses
        # cross-referencing entirely — the target renders as plain inline
        # code, no lookup, no warning. Sphinx strips it *before* the
        # "Title <target>" split, so check here and again on the target
        # after the split.
        suppress = False
        if to_resolve.startswith("!"):
            suppress = True
            text = text[1:]
            to_resolve = to_resolve[1:]

        if (
            ("<" in text)
            and text.endswith(">")
            and " <" not in text
            and "\n<" not in text
        ):
            pass  # assert False, ("error space-< in", self.qa, directive)
        if ((" <" in text) and text.endswith(">")) or (
            ("\n <" in text) and text.endswith(">")
        ):
            try:
                text, to_resolve = text.split(" <", 1)
                text = text.rstrip()
            except ValueError as e:
                raise AssertionError(directive.value) from e
            assert to_resolve.endswith(">"), (text, to_resolve)
            to_resolve = to_resolve.rstrip(">")

        if to_resolve.startswith("!"):
            suppress = True
            stripped = to_resolve[1:]
            if text == to_resolve:
                text = stripped
            to_resolve = stripped

        if to_resolve.startswith("~"):
            stripped = to_resolve[1:]
            if text == to_resolve:
                text = stripped.split(".")[-1]
            to_resolve = stripped

        if suppress:
            return [InlineCode(text)]

        if to_resolve.startswith(("https://", "http://", "mailto://")):
            to_resolve = to_resolve.replace(" ", "")
            return [
                Link(
                    children=[Text(text)],
                    url=to_resolve,
                    title="",
                )
            ]

        # :ref:`label` — RST cross-reference to a named target within the bundle.
        # Resolved against doc_targets collected during the first parse pass.
        # domain has already been remapped to "py" for None-domain roles so we
        # match on role alone — "ref" is unambiguous across domains.
        if role == "ref":
            label = to_resolve
            if label in self.doc_targets:
                doc_key = self.doc_targets[label]
                return [CrossRef(text, LocalRef("docs", doc_key), "exists")]
            else:
                self.diagnostics.emit(
                    W_UNRESOLVED_REF,
                    self.qa,
                    f"unresolved :ref: label {label!r}",
                )
                return [directive]

        # :doc:`path` — RST cross-reference to another document. Paths use
        # "/" separators ("/" prefix anchors at the docs root, otherwise
        # relative to the current document); normalize to the ":"-joined doc
        # key so the LocalRef matches how narrative docs are stored.
        if role == "doc":
            doc_path = to_resolve
            in_api_docstring = (
                self.qa == self.module
                or self.qa.startswith(self.module + ".")
                or self.qa.startswith(self.module + ":")
            )
            if in_api_docstring and not doc_path.startswith("/"):
                # API objects are not documents — there is no "current
                # document directory" to resolve against, so anchor
                # relative :doc: paths at the docs root.
                doc_path = "/" + doc_path
            doc_key = self._resolve_doc_path(doc_path)
            display = text if text != to_resolve else self.doc_titles.get(doc_key, text)
            return [CrossRef(display, LocalRef("docs", doc_key), "docs")]

        # Plain RST hyperlink with angle-bracket syntax (``text <label>`_``) or
        # bare named reference (``label_``) where the target matches a known
        # doc anchor or recorded external URL. role is None here (remapped to
        # "py" above) so the :ref: branch never fires for these.
        #
        # The bare ``label_`` form keeps its trailing underscore in the value,
        # so probe both the raw label and the stripped form when looking up
        # targets. For the display text, use the explicit text from the
        # angle-bracket form, or the stripped label otherwise.
        if directive.role is None:
            if to_resolve.endswith("__"):
                candidates = [to_resolve, to_resolve[:-2]]
            elif to_resolve.endswith("_"):
                candidates = [to_resolve, to_resolve[:-1]]
            else:
                candidates = [to_resolve]

            for cand in candidates:
                display = cand if text == to_resolve else text
                if cand in self.doc_targets:
                    return [
                        CrossRef(
                            display,
                            LocalRef("docs", self.doc_targets[cand]),
                            "exists",
                        )
                    ]
                if cand in self.external_targets:
                    return [
                        Link(
                            children=[Text(display)],
                            url=self.external_targets[cand],
                            title="",
                        )
                    ]

        # Sphinx py roles ignore a trailing pair of parentheses on the target
        # (":meth:`foo()`" links to foo); the display text keeps them.
        if to_resolve.endswith("()"):
            to_resolve = to_resolve[:-2]

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
            return [self._ref_to_crossref(text, r, exists)]
        # Python-object roles fall back to the import solver when the
        # in-bundle lookup above produced no match.
        if directive.role in _PYTHON_OBJECT_ROLES and directive.domain in (None, "py"):
            text = directive.value
            tqa = directive.value

            if text.startswith("@"):
                tqa = tqa[1:]
            if text.startswith("~"):
                tqa = tqa[1:]
                text = tqa.split(".")[-1]
            # Sphinx convention: a leading "." makes the reference relative
            # to the current module (e.g. ".foo" inside numpy resolves to
            # "numpy.foo"). Previously we just stripped the dot, which left
            # the lookup unqualified and almost always failed to resolve.
            if tqa.startswith("."):
                tqa = self.module + tqa
            if tqa.endswith("()"):
                tqa = tqa[:-2]

            target_qa = self._import_solver(tqa)
            if target_qa is not None:
                module = target_qa.split(":")[0].split(".")[0]
                # Emit the canonical cross-package form (kind="module",
                # version="?") directly. Since the sentinel unification the
                # consumer side no longer normalises the old (kind="api",
                # version="*") form, so producing it here would strand the
                # edge under a phantom node category that no stored page uses.
                ri = RefInfo(
                    module=module,
                    version="?",
                    kind="module",
                    path=target_qa,
                )
                return [self._ref_to_crossref(text, ri, "module")]
        role_desc = directive.role or "(default)"
        # Bare backticks (no explicit role) are routinely used for variable
        # names; Sphinx's autolink default role degrades to plain text
        # silently, so an unresolved default role gets its own, quieter code.
        code = (
            W_UNRESOLVED_REF
            if directive.role is not None
            else W_UNRESOLVED_DEFAULT_ROLE
        )
        self.diagnostics.emit(
            code,
            self.qa,
            f"unresolved reference {directive.value!r} (role {role_desc})",
        )
        return [directive]


def _import_max(parts: list[str]) -> None:
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
            raise type(e)(parts) from e


def _obj_from_path(parts: list[str]) -> Any:
    _import_max(parts)
    try:
        target = __import__(parts[0])
        for p in parts[1:]:
            target = getattr(target, p)
    except Exception:
        return
    return target


class GenVisitor(DirectiveVisiter):
    def visit_Section(self, node: Any) -> None:
        if node.target:
            RESOLVER.add_target(LocalRef("docs", node.target), node.target)

    def replace_Fig(self, fig: Any) -> list[Any]:
        # todo: add version number here
        self._targets.add(fig.value)

        return [fig]

    def _ref_to_crossref(self, text: str, r: RefInfo, exists: str) -> CrossRef:
        # Intra-bundle refs don't need a version stamp — store as LocalRef so
        # the bundle digest is independent of its own version number.
        if r.module == self.module:
            self._targets.discard(r)
            return CrossRef(text, LocalRef(r.kind, r.path), exists)
        return CrossRef(text, r, exists)
