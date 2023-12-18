"""
Main module responsible from scrapping the code, docstrings, an (TBD) rst files,
and turning that into intermediate representation files that can be published.

This also does some code execution, and inlining of figures, though that should
likely be separated into a separate module at some point.


"""

from __future__ import annotations

import doctest
import dataclasses
import datetime
import inspect
import json
import logging
import os
import shutil
import site
import sys
import tempfile
import tomllib
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import count
from pathlib import Path
from types import FunctionType, ModuleType
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import io

import jedi
import tomli_w
from IPython.core.oinspect import find_file
from IPython.utils.path import compress_user
from packaging.version import parse
from pygments import lex
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, track
from there import print as print_
from matplotlib import _pylab_helpers

from .common_ast import Node
from .errors import (
    IncorrectInternalDocsLen,
    NumpydocParseError,
    TextSignatureParsingFailed,
    UnseenError,
)
from .miscs import BlockExecutor, DummyP
from .signature import Signature as ObjectSignature
from .signature import SignatureNode
from .take2 import (
    Code,
    Fig,
    GenToken,
    Link,
    NumpydocExample,
    NumpydocSeeAlso,
    NumpydocSignature,
    Param,
    Parameters,
    RefInfo,
    Section,
    SeeAlsoItem,
    SubstitutionDef,
    parse_rst_section,
)
from .toc import make_tree
from .tree import DVR
from .utils import (
    Cannonical,
    FullQual,
    TimeElapsedColumn,
    dedent_but_first,
    full_qual,
    pos_to_nl,
    progress,
    obj_from_qualname,
)
from .vref import NumpyDocString

# delayed import
if True:
    from .myst_ast import MText


class ErrorCollector:
    _expected_unseen: Dict[str, Any]
    errored: bool
    _unexpected_errors: Dict[str, Any]
    _expected_errors: Dict[str, Any]

    def __init__(self, config: Config, log):
        self.config: Config = config
        self.log = log

        self._expected_unseen = {}
        for err, names in self.config.expected_errors.items():
            for name in names:
                self._expected_unseen.setdefault(name, []).append(err)
        self._unexpected_errors = {}
        self._expected_errors = {}

    def __call__(self, qa):
        self._qa = qa
        return self

    def __enter__(self):
        self.errored = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type in (BaseException, KeyboardInterrupt):
            return
        if exc_type:
            self.errored = True
            ename = exc_type.__name__
            if ename in self._expected_unseen.get(self._qa, []):
                self._expected_unseen[self._qa].remove(ename)
                if not self._expected_unseen[self._qa]:
                    del self._expected_unseen[self._qa]
                self._expected_errors.setdefault(ename, []).append(self._qa)
            else:
                self._unexpected_errors.setdefault(ename, []).append(self._qa)
                self.log.exception(f"Unexpected error {self._qa}")
            if not self.config.early_error:
                return True
        expecting = self._expected_unseen.get(self._qa, [])
        if expecting and self.config.fail_unseen_error:
            raise UnseenError(f"Expecting one of {expecting}")

        # return True


try:
    from . import ts
except (ImportError, OSError):
    sys.exit(
        """
            Tree Sitter RST parser not available, you may need to:

            $ git clone https://github.com/stsewd/tree-sitter-rst
            $ papyri build-parser
            """
    )
SITE_PACKAGE = site.getsitepackages()


def paragraph(lines: List[str], qa) -> Any:
    """
    Leftover rst parsing,

    Remove at some point.
    """
    [section] = ts.parse("\n".join(lines).encode(), qa)
    assert len(section.children) == 1
    p2 = section.children[0]
    return p2


_JEDI_CACHE = Path("~/.cache/papyri/jedi/").expanduser()


def _hashf(text):
    ##  for cache expiring every day.
    ## for every hours, change to 0:13.

    return sha256(text.encode()).hexdigest() + datetime.datetime.now().isoformat()[0:10]


def _jedi_get_cache(text):
    _JEDI_CACHE.mkdir(exist_ok=True, parents=True)

    _cache = _JEDI_CACHE / _hashf(text)
    if _cache.exists():
        return tuple(tuple(x) for x in json.loads(_cache.read_text()))

    return None


def _jedi_set_cache(text, value):
    _JEDI_CACHE.mkdir(exist_ok=True, parents=True)

    _cache = _JEDI_CACHE / _hashf(text)
    _cache.write_text(json.dumps(value))


def parse_script(
    script: str, ns: Dict, prev, config, *, where=None
) -> Optional[List[Tuple[str, Optional[str]]]]:
    """
    Parse a script into tokens and use Jedi to infer the fully qualified names
    of each token.

    Parameters
    ----------
    script : str
        the script to tokenize and infer types on
    ns : dict
        Extra namespace to use with jedi's Interpreter. This will be used for
        implicit imports, for example that `np` is interpreted as numpy.
    prev : str
        previous lines that lead to this.
    where : <Insert Type here>
        <Multiline Description Here>
    config : <Insert Type here>
        <Multiline Description Here>

    Returns
    -------
    List of tuples with:
    text:
        text of the token
    reference : str
        fully qualified name of the type of current token

    """
    assert isinstance(ns, dict)
    jeds = []
    warnings.simplefilter("ignore", UserWarning)

    l_delta = len(prev.split("\n"))
    contextscript = prev + "\n" + script
    if ns:
        jeds.append(jedi.Interpreter(contextscript, namespaces=[ns]))
    full_text = prev + "\n" + script
    k = _jedi_get_cache(full_text)
    if k is not None:
        return k
    jeds.append(jedi.Script(full_text))
    P = PythonLexer()

    acc: List[Tuple[str, Optional[str]]] = []

    for index, _type, text in P.get_tokens_unprocessed(script):
        line_n, col_n = pos_to_nl(script, index)
        line_n += l_delta
        ref = None
        if not config.infer or (text in (" .=()[],")) or not text.isidentifier():
            acc.append((text, ""))
            continue

        for jed in jeds:
            try:
                inf = jed.infer(line_n + 1, col_n)
                if inf:
                    # TODO: we might want the qualname to
                    # be module_name:name for disambiguation.
                    ref = inf[0].full_name
            except (AttributeError, TypeError) as e:
                raise type(e)(
                    f"{contextscript}, {line_n=}, {col_n=}, {prev=}, {jed=}"
                ) from e
            except jedi.inference.utils.UncaughtAttributeError:
                if config.jedi_failure_mode in (None, "error"):
                    raise
                elif config.jedi_failure_mode == "log":
                    print_(
                        "failed inference example will be empty ",
                        where,
                        line_n,
                        col_n,
                    )
                    return None
            break
        acc.append((text, ref))
    _jedi_set_cache(full_text, acc)
    warnings.simplefilter("default", UserWarning)
    for a in acc:
        assert len(a) == 2
    return acc


from enum import Enum


class ExecutionStatus(Enum):
    success = "success"
    failure = "failure"
    unexpected_exception = "unexpected_exception"


def _execute_inout(item):
    script = "\n".join(item.in_)
    ce_status = ExecutionStatus.none
    try:
        compile(script, "<>", "exec")
        ce_status = ExecutionStatus.compiled
    except SyntaxError:
        ce_status = ExecutionStatus.syntax_error

    return script, item.out, ce_status.value


def _get_implied_imports(obj):
    """
    Most examples in methods or modules needs names defined in current module,
    or name of the class they are part of.
    """
    if hasattr(obj, "__qualname__"):
        if "." not in obj.__qualname__:
            return {}
        else:
            c_o = obj.__qualname__.split(".")
            if len(c_o) > 2:
                print_(
                    "get implied import qualname got more than 2 parts: ",
                    obj.__qualname__,
                )
                return {}
            cname, oname = c_o
            mod_name = obj.__module__
            import importlib

            mod = importlib.import_module(mod_name)
            return {cname: getattr(mod, cname)}

    return {}


def get_classes(code):
    """
    Extract Pygments token classes names for given code block
    """
    list(lex(code, PythonLexer()))
    FMT = HtmlFormatter()
    classes = [FMT.ttype2class.get(x) for x, y in lex(code, PythonLexer())]
    classes = [c if c is not None else "" for c in classes]
    return classes


def _add_classes(entries):
    assert set(len(x) for x in entries) == {2}
    text = "".join([x for x, y in entries])
    classes = get_classes(text)
    return [ii + (cc,) for ii, cc in zip(entries, classes)]


def processed_example_data(example_section_data) -> Section:
    """this should be no-op on already ingested"""
    new_example_section_data = Section([], None)
    for in_out in example_section_data:
        type_ = in_out.__class__.__name__
        # color examples with pygments classes
        if type_ == "Text":
            assert False

        if type_ != "Text":
            new_example_section_data.append(in_out)
    return new_example_section_data


@lru_cache
def normalise_ref(ref):
    """
    Consistently normalize references.

    Refs are sometime import path, not fully qualified names, tough type
    inference in examples regularly give us fully qualified names. When visiting
    a ref, this tries to import it and replace it by the normal full-qualified form.

    This is expensive, ad we likely want to move the logic of finding the
    correct ref earlier in the process and us this as an assertion the refs are
    normalized.

    It is critical to normalize in order to have the correct information when
    using interactive ?/??, or similar inspector of live objects;

    """
    if ref.startswith(("builtins.", "__main__")):
        return ref
    try:
        mod_name, name = ref.rsplit(".", maxsplit=1)
        mod = __import__(mod_name)
        for sub in mod_name.split(".")[1:]:
            mod = getattr(mod, sub)
        obj = getattr(mod, name)
        if isinstance(obj, ModuleType):
            return ref
        if getattr(obj, "__name__", None) is None:
            return ref

        return obj.__module__ + "." + obj.__name__
    except Exception:
        pass
    return ref


@dataclass
class Config:
    # we might want to suppress progress/ rich as it infers with ipdb.
    dummy_progress: bool = False
    # Do not actually touch disk
    dry_run: bool = False
    exec_failure: Optional[str] = None  # should move to enum
    jedi_failure_mode: Optional[str] = None  # move to enum ?
    logo: Optional[str] = None  # should change to path likely
    execute_exclude_patterns: Sequence[str] = ()
    infer: bool = True
    exclude: Sequence[str] = ()  # list of dotted object name to exclude from collection
    examples_folder: Optional[str] = None  # < to path ?
    submodules: Sequence[str] = ()
    source: Optional[str] = None
    homepage: Optional[str] = None
    docs: Optional[str] = None
    docs_path: Optional[str] = None
    wait_for_plt_show: Optional[bool] = True
    examples_exclude: Sequence[str] = ()
    narrative_exclude: Sequence[str] = ()
    exclude_jedi: Sequence[str] = ()
    implied_imports: Dict[str, str] = dataclasses.field(default_factory=dict)
    # mapping from expected name of error instances, to which fully-qualified names are raising those errors.
    # the build will fail if the given item does not raise this error.
    expected_errors: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    early_error: bool = True
    fail_unseen_error: bool = False
    execute_doctests: bool = True
    directives: Dict[str, str] = dataclasses.field(default_factory=lambda: {})

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def load_configuration(
    path: str,
) -> Tuple[str, MutableMapping[str, Any], Dict[str, Any]]:
    """
    Given a path, load a configuration from a File.

    Each configuration file should have two sections: ['global', 'meta'] where
    the name of the module should be defined under the 'global' section.
    Additionally, a section for expected errors can be defined.
    """
    conffile = Path(path).expanduser()
    if conffile.exists():
        conf: MutableMapping[str, Any] = tomllib.loads(conffile.read_text())
        ks = set(conf.keys()) - {"meta"}
        assert len(ks) >= 1, conf.keys()
        info = conf["global"]
        root = info.pop("module")
        return root, info, conf.get("meta", {})
    else:
        sys.exit(f"{conffile!r} does not exist.")


def gen_main(
    infer: Optional[bool],
    exec_: Optional[bool],
    target_file: str,
    debug,
    *,
    dummy_progress: bool,
    dry_run: bool,
    api: bool,
    examples: bool,
    fail,
    narrative,
    fail_early: bool,
    fail_unseen_error: bool,
    limit_to: List[str],
) -> None:
    """
    Main entry point to generate DocBundle files.

    This will take care of reading single configuration files with the options
    for the library you want to build the docs for, scrape API, narrative and
    examples, and put it into a doc bundle for later consumption.

    Parameters
    ----------
    infer : bool | None
        CLI override of whether to run type inference on examples
    exec_ : bool | None
        CLI override of whether to execute examples/code blocks
    target_file : str
        Patch of configuration file
    dummy_progress : bool
        CLI flag to disable progress that might screw up with ipdb formatting
        when debugging.
    api : bool
        CLI override of whether to build api docs
    examples : bool
        CLI override of whether to build examples docs
    fail
        TBD
    narrative : bool
        CLI override of whether to build narrative docs
    dry_run : bool
        don't write to disk
    debug : bool
        set log level to debug
    fail_early : bool
        overwrite early_error option in config file
    fail_unseen_error : bool
        raise an exception if the error is unseen

    Returns
    -------
    None

    """
    if limit_to is None:
        limit_to = set()
    target_module_name, conf, meta = load_configuration(target_file)

    conf["early_error"] = fail_early
    conf["fail_unseen_error"] = fail_unseen_error
    config = Config(**conf, dry_run=dry_run, dummy_progress=dummy_progress)
    if exec_ is not None:
        config.execute_doctests = exec_
    if infer is not None:
        config.infer = infer

    target_dir = Path("~/.papyri/data").expanduser()

    if not target_dir.exists() and not config.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        temp_dir = tempfile.TemporaryDirectory()
        target_dir = Path(temp_dir.name)

    g = Gen(dummy_progress=dummy_progress, config=config)

    if debug:
        g.log.setLevel(logging.DEBUG)
        g.log.debug("Log level set to debug")

    g.collect_package_metadata(
        target_module_name,
        relative_dir=Path(target_file).parent,
        meta=meta,
    )

    g.log.info("Target package is %s-%s", target_module_name, g.version)
    g.log.info("Will write data to %s", target_dir)

    if examples:
        g.collect_examples_out()
    if api:
        g.collect_api_docs(target_module_name, limit_to=limit_to)
    if narrative:
        g.collect_narrative_docs()

    p = target_dir / (g.root + "_" + g.version)
    p.mkdir(exist_ok=True)

    g.log.info("Saving current Doc bundle to %s", p)
    if not limit_to:
        g.clean(p)
        g.write(p)
    else:
        g.partial_write(p)
    if dry_run:
        temp_dir.cleanup()


def pack():
    target_dir = Path("~/.papyri/data").expanduser()
    dirs = [d for d in target_dir.glob("*") if d.is_dir()]
    for d in track(dirs, description=f"packing {len(dirs)} items..."):
        shutil.make_archive(d, "zip", d)


class DFSCollector:
    """
    Depth first search collector.

    Will scan documentation to find all reachable items in the namespace
    of our root object (we don't want to go scan other libraries).

    Three was some issues with BFS collector originally, I'm not sure I remember what.


    """

    def __init__(self, root: ModuleType, others: List[ModuleType]):
        """
        Parameters
        ----------
        root
            Base object, typically module we want to scan itself.
            We will attempt to no scan any object which does not belong
            to the root or one of its children.
        others
            List of other objects to use a base to explore the object graph.
            Typically this is because some packages do not import some
            submodules by default, so we need to pass these submodules
            explicitly.
        """

        assert isinstance(root, ModuleType), root
        self.root = root.__name__
        assert "." not in self.root
        self.obj: Dict[str, Any] = dict()
        self.aliases: Dict[str, List[str]] = defaultdict(lambda: [])
        self._open_list = [(root, [root.__name__])]
        for o in others:
            self._open_list.append((o, o.__name__.split(".")))
        self.log = logging.getLogger("papyri")

    def scan(self) -> None:
        """
        Attempt to find all objects.
        """
        while len(self._open_list) >= 1:
            current, stack = self._open_list.pop(0)

            # numpy objects ane no bool values.
            if id(current) not in [id(x) for x in self.obj.values()]:
                self.visit(current, stack)

    def prune(self) -> None:
        """
        Some object can be reached many times via multiple path.
        We try to remove duplicate path we use to reach given objects.

        Notes
        -----
        At some point we might want to save all objects aliases,
        in order to extract the canonical import name (visible to users),
        and to resolve references.
        """
        for qa, item in self.obj.items():
            if (nqa := full_qual(item)) != qa:
                print_("after import qa differs : {qa} -> {nqa}")
                assert isinstance(nqa, str)
                if self.obj[nqa] == item:
                    print_("present twice")
                    del self.obj[nqa]
                else:
                    print_("differs: {item} != {other}")

    def items(self) -> Dict[str, Any]:
        self.scan()
        self.prune()
        return self.obj

    def visit(self, obj, stack):
        """
        Recursively visit Module, Classes, and Functions by tracking which path
        we took there.
        """

        try:
            qa = full_qual(obj)
        except Exception as e:
            raise RuntimeError(f"error visiting {'.'.join(self.stack)}") from e
        if not qa:
            if (
                "__doc__" not in stack
                and hasattr(obj, "__doc__")
                and not full_qual(type(obj)).startswith("builtins.")
            ):
                # might be worth looking into like np.exp.
                pass
            return

        if ":" in qa:
            omod, _name = qa.split(":")
        else:
            omod = qa

        if "." in omod:
            oroot = omod.split(".")[0]
        else:
            oroot = omod

        if oroot != self.root:
            return
        if obj in self.obj.values():
            return
        if (qa in self.obj) and self.obj[qa] != obj:
            pass

        self.obj[qa] = obj
        self.aliases[qa].append(".".join(stack))

        if isinstance(obj, ModuleType):
            return self.visit_ModuleType(obj, stack)
        elif isinstance(obj, FunctionType):
            return self.visit_FunctionType(obj, stack)
        elif isinstance(obj, type):
            return self.visit_ClassType(obj, stack)
        else:
            pass

    def visit_ModuleType(self, mod, stack):
        for k in dir(mod):
            # TODO: scipy 1.8 workaround, remove.
            if not hasattr(mod, k):
                self.log.warning(f"Name not found in module: {mod.__name__}.{k}")
                continue
            self._open_list.append((getattr(mod, k), stack + [k]))

    def visit_ClassType(self, klass, stack):
        for k, v in klass.__dict__.items():
            self._open_list.append((v, stack + [k]))

    def visit_FunctionType(self, fun, stack):
        pass

    def compute_aliases(self) -> Tuple[Dict[FullQual, Cannonical], List[Any]]:
        aliases = {}
        not_found = []
        for k, v in self.aliases.items():
            if [item for item in v if item != k]:
                if shorter := find_cannonical(k, v):
                    aliases[FullQual(k)] = Cannonical(shorter)
                else:
                    not_found.append((k, v))
        return aliases, not_found


class _OrderedDictProxy:
    """
    a dict like class proxy for DocBlob to keep the order of sections in DocBlob.

    We Can't use an ordered Dict because of serialisation/deserialisation that
    would/might loose order
    """

    orderring: list[str]
    mapping: dict[str, Any]

    def __init__(self, ordering: list[str], mapping: dict[str, Any]):
        self.ordering = ordering
        self.mapping = mapping
        assert isinstance(ordering, list), ordering
        assert isinstance(mapping, dict), mapping
        assert set(self.mapping.keys()) == set(self.ordering)

    def __getitem__(self, key: str):
        return self.mapping[key]

    def __contains__(self, key: str):
        return key in self.mapping

    def __setitem__(self, key: str, value: Any):
        if key not in self.ordering:
            self.ordering.append(key)
        self.mapping[key] = value

    def __delitem__(self, key: str):
        self.ordering.remove(key)
        del self.mapping[key]

    def __iter__(self):
        return iter(self.ordering)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.ordering)

    def get(self, key: str, default=None, /):
        return self.mapping.get(key, default)

    def items(self):
        return [(k, self.mapping[k]) for k in self.ordering]

    def values(self):
        return [self.mapping[k] for k in self.ordering]


class DocBlob(Node):
    """
    An object containing information about the documentation of an arbitrary
    object.

    Instead of DocBlob being a NumpyDocString, I'm thinking of them having a
    NumpyDocString. This helps with arbitrary documents (module, examples files)
    that cannot be parsed by Numpydoc, as well as links to external references,
    like images generated.

    """

    __slots__ = (
        "_content",
        "example_section_data",
        "_ordered_sections",
        "item_file",
        "item_line",
        "item_type",
        "aliases",
        "see_also",
        "signature",
        "references",
        "arbitrary",
        "_dp",
    )

    @classmethod
    def _deserialise(cls, **kwargs):
        # print_("will deserialise", cls)
        try:
            instance = cls(**kwargs)
        except Exception as e:
            raise type(e)(f"Error deserialising {cls}, {kwargs})") from e
        for k, v in kwargs.items():
            setattr(instance, k, v)
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not isinstance(self._content, str)
        self._dp = _OrderedDictProxy(self._ordered_sections, self._content)

    @property
    def ordered_sections(self):
        return tuple(self._ordered_sections)

    @property
    def content(self):
        return self._dp

    sections = [
        "Signature",
        "Summary",
        "Extended Summary",
        "Parameters",
        "Returns",
        "Yields",
        "Receives",
        "Raises",
        "Warns",
        "Other Parameters",
        "Attributes",
        "Methods",
        "See Also",
        "Notes",
        "Warnings",
        "References",
        "Examples",
    ]  # List of sections in order

    _content: Dict[str, Section]
    example_section_data: Section
    _ordered_sections: Optional[List[str]]
    item_file: Optional[str]
    item_line: Optional[int]
    item_type: Optional[str]
    aliases: List[str]
    see_also: List[SeeAlsoItem]  # see also data
    signature: Optional[SignatureNode]
    references: Optional[List[str]]
    arbitrary: List[Section]

    def __repr__(self):
        return "<DocBlob ...>"

    def slots(self):
        return [
            "_content",
            "example_section_data",
            "_ordered_sections",
            "item_file",
            "item_line",
            "item_type",
            "signature",
            "aliases",
            "arbitrary",
        ]

    @classmethod
    def new(cls):
        return cls({}, None, [], None, None, None, [], [], None, None, [])


def _numpy_data_to_section(data: List[Tuple[str, str, List[str]]], title: str, qa):
    assert isinstance(data, list), repr(data)
    acc = []
    for param, type_, desc in data:
        assert isinstance(desc, list)
        items = []
        if desc:
            items = parse_rst_section("\n".join(desc), qa)
            for l in items:
                assert not isinstance(l, Section)
        acc.append(Param(param, type_, desc=items).validate())
    if acc:
        return Section([Parameters(acc)], title).validate()
    else:
        return Section([], title)


_numpydoc_sections_with_param = {
    "Parameters",
    "Returns",
    "Raises",
    "Yields",
    "Attributes",
    "Other Parameters",
    "Warns",
    "Methods",
    "Receives",
}

_numpydoc_sections_with_text = {
    "Summary",
    "Notes",
    "Extended Summary",
    "References",
    "Warnings",
}


class APIObjectInfo:
    """
    Describes the object's type and other relevant information

    This object can be many things, such as a Module, Class, method, function.

    """

    kind: str
    docstring: str
    signature: Optional[ObjectSignature]
    name: str

    def __repr__(self):
        return f"<APIObject {self.kind=} {self.docstring=} self.signature={str(self.signature)} {self.name=}>"

    def __init__(
        self,
        kind: str,
        docstring: str,
        signature: Optional[ObjectSignature],
        name: str,
        qa: str,
    ):
        assert isinstance(signature, (ObjectSignature, type(None)))
        self.kind = kind
        self.name = name
        self.docstring = docstring
        self.parsed: List[Any] = []
        self.signature = signature
        self._qa = qa

        if docstring is not None and kind != "module":
            # TS is going to choke on this as See Also and other
            # sections are technically invalid.
            try:
                ndoc = NumpyDocString(dedent_but_first(docstring))
            except Exception as e:
                raise NumpydocParseError("APIObjectInfoParse Error in numpydoc") from e

            for title in ndoc.ordered_sections:
                if not ndoc[title]:
                    continue
                if title in _numpydoc_sections_with_param:
                    section = _numpy_data_to_section(ndoc[title], title, self._qa)
                    assert isinstance(section, Section)
                    self.parsed.append(section)
                elif title in _numpydoc_sections_with_text:
                    predoc = "\n".join(ndoc[title])
                    docs = ts.parse(predoc.encode(), qa)
                    if len(docs) != 1:
                        # TODO
                        # potential reasons
                        # Summary and Extended Summary should be parsed as one.
                        # References with ` : ` in them fail parsing.Issue opened in Tree-sitter.
                        raise IncorrectInternalDocsLen(predoc, docs)
                    section = docs[0]
                    assert isinstance(section, Section), section
                    self.parsed.append(section)
                elif title == "Signature":
                    self.parsed.append(NumpydocSignature(ndoc[title]))
                elif title == "Examples":
                    self.parsed.append(NumpydocExample(ndoc[title]))
                elif title == "See Also":
                    see_also = ndoc[title]
                    xx = NumpydocSeeAlso(_normalize_see_also(see_also, qa="??"))
                    self.parsed.append(xx)
                else:
                    assert False
        elif docstring and kind == "module":
            self.parsed = ts.parse(docstring.encode(), qa)
        self.validate()

    def special(self, title):
        if self.kind == "module":
            return None
        res = [s for s in self.parsed if s.title == title]
        if not res:
            return None
        assert len(res) == 1
        assert not isinstance(res[0], Section), self.parsed
        return res[0]

    def validate(self):
        for p in self.parsed:
            assert isinstance(
                p, (Section, NumpydocExample, NumpydocSeeAlso, NumpydocSignature)
            )
            p.validate()


def _normalize_see_also(see_also: Section, qa: str):
    """
    numpydoc is complex, the See Also fields can be quite complicated,
    so here we sort of try to normalise them.
    from what I can remember,
    See also can have
    name1 : type1
    name2 : type2
        description for both name1 and name 2.

    Though if description is empty, them the type is actually the description.
    """
    if not see_also:
        return []
    assert see_also is not None
    new_see_also = []
    name_and_types: List[Tuple[str, str]]
    name: str
    type_or_description: str

    for name_and_types, raw_description in see_also:
        try:
            for name, type_or_description in name_and_types:
                if type_or_description and not raw_description:
                    assert isinstance(type_or_description, str)
                    type_ = None
                    # we have all in a single line,
                    # and there is no description, so the type field is
                    # actually the description.
                    desc = [paragraph([type_or_description], qa)]
                elif raw_description:
                    assert isinstance(raw_description, list)
                    type_ = type_or_description
                    desc = [paragraph(raw_description, qa)]
                else:
                    type_ = type_or_description
                    desc = []
                refinfo = RefInfo.from_untrusted(
                    "current-module", "current-version", "to-resolve", name
                )
                link = Link(name, refinfo, "module", True)
                sai = SeeAlsoItem(link, desc, type_)
                new_see_also.append(sai)
                del desc
                del type_
        except Exception as e:
            raise ValueError(
                f"Error {qa}: {see_also=} | {name_and_types=}  | {raw_description=}"
            ) from e
    return new_see_also


class PapyriDocTestRunner(doctest.DocTestRunner):
    def __init__(self, *args, gen, obj, qa, config, **kwargs):
        self._count = count(0)
        self.gen = gen
        self.obj = obj
        self.qa = qa
        self.config = config
        self._example_section_data = Section([], None)
        super().__init__(*args, **kwargs)
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

        matplotlib.use("agg")

        self.globs = {"np": np, "plt": plt, obj.__name__: obj}
        self.globs.update(_get_implied_imports(obj))
        for k, v in config.implied_imports.items():
            self.globs[k] = obj_from_qualname(v)

        self.figs = []

    def _get_tok_entries(self, example):
        entries = parse_script(
            example.source, ns=self.globs, prev="", config=self.config, where=self.qa
        )
        if entries is None:
            entries = [("jedi failed", "jedi failed")]
        entries = _add_classes(entries)
        tok_entries = [GenToken(*x) for x in entries]  # type: ignore
        return tok_entries

    def _next_figure_name(self):
        """
        File system can be case insensitive, we are not.
        """
        i = next(self._count)
        pat = f"fig-{self.qa}-{i}"
        sha = sha256(pat.encode()).hexdigest()[:8]
        return f"{pat}-{sha}.png"

    def report_start(self, out, test, example):
        pass

    def report_success(self, out, test, example, got):
        import matplotlib.pyplot as plt

        tok_entries = self._get_tok_entries(example)

        self._example_section_data.append(
            Code(tok_entries, got, ExecutionStatus.success)
        )

        wait_for_show = self.config.wait_for_plt_show
        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
        figs = []
        if fig_managers and (("plt.show" in example.source) or not wait_for_show):
            for fig in fig_managers:
                figname = self._next_figure_name()
                buf = io.BytesIO()
                fig.canvas.figure.savefig(buf, dpi=300)  # , bbox_inches="tight"
                buf.seek(0)
                figs.append((figname, buf.read()))
            plt.close("all")

        for figname, _ in figs:
            self._example_section_data.append(
                Fig(
                    RefInfo.from_untrusted(
                        self.gen.root, self.gen.version, "assets", figname
                    )
                )
            )
        self.figs.extend(figs)

    def report_unexpected_exception(self, out, test, example, exc_info):
        out(f"Unexpected exception after running example in `{self.qa}`", exc_info)
        tok_entries = self._get_tok_entries(example)
        self._example_section_data.append(
            Code(tok_entries, exc_info, ExecutionStatus.unexpected_exception)
        )

    def report_failure(self, out, test, example, got):
        tok_entries = self._get_tok_entries(example)
        self._example_section_data.append(
            Code(tok_entries, got, ExecutionStatus.failure)
        )

    def get_example_section_data(self) -> Section:
        example_section_data = self._example_section_data
        self._example_section_data = Section([], None)
        return example_section_data

    def _compact(self, example_section_data) -> Section:
        """
        Compact consecutive execution items that do have the same execution status.

        TODO:

        This is not perfect as doctest tests that the output is the same, thus when we have a multiline block
        If any of the intermediate items produce an output, the result will be failure.
        """
        acc: List[Union[MText, Code]] = []
        current_code: Optional[Code] = None

        for item in example_section_data:
            if not isinstance(item, Code):
                if current_code is not None:
                    acc.append(current_code)
                    acc.append(MText(str(current_code.out)))
                    acc.append(MText(str(current_code.ce_status)))
                    current_code = None
                acc.append(item)
            else:
                if current_code is None:
                    assert item is not None
                    current_code = item
                    continue

                if current_code.ce_status == item.ce_status:
                    current_code = Code(
                        current_code.entries + item.entries, item.out, item.ce_status
                    )
                else:
                    acc.append(current_code)
                    acc.append(MText(str(current_code.out)))
                    acc.append(MText(str(current_code.ce_status)))
                    assert item is not None
                    current_code = item

        if current_code:
            acc.append(current_code)
        return Section(acc, None)


class Gen:
    """
    Core class to generate a DocBundle for a given library.

    This is responsible for finding all objects, extracting the doc, parsing it,
    and saving that into the right folder.

    """

    docs: Dict[str, bytes]
    examples: Dict[str, bytes]
    data: Dict[str, DocBlob]
    bdata: Dict[str, bytes]

    def __init__(self, dummy_progress: bool, config: Config):
        if dummy_progress:
            self.Progress = DummyP
        else:
            self.Progress = Progress  # type: ignore

        self.progress = lambda: self.Progress(
            TextColumn("[progress.description]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "[progress.completed]{task.completed} / {task.total}",
            TimeElapsedColumn(),
        )

        # TODO:
        # At some point it would be better to have that be matplotlib
        # specific and not hardcoded.
        class MF(logging.Filter):
            """
            This is a matplotlib filter to temporarily silence a bunch of warning
            messages that are emitted if font are not found

            """

            def filter(self, record):
                if "Generic family" in record.msg:
                    return 0
                if "found for the serif fontfamily" in record.msg:
                    return 0
                if "not found. Falling back to" in record.msg:
                    return 0
                if "Substituting symbol" in record.msg:
                    return 0
                return 1

        mlog = logging.getLogger("matplotlib.font_manager")
        mlog.addFilter(MF("serif"))

        mplog = logging.getLogger("matplotlib.mathtext")
        mplog.addFilter(MF("serif"))

        # end TODO

        FORMAT = "%(message)s"
        self.log = logging.getLogger("papyri")
        self.log.setLevel("INFO")
        formatter = logging.Formatter(FORMAT, datefmt="[%X]")
        rich_handler = RichHandler(rich_tracebacks=False)
        rich_handler.setFormatter(formatter)
        self.log.addHandler(rich_handler)

        self.config = config
        self.log.debug("Configuration: %s", self.config)

        self.data = {}
        self.bdata = {}
        self._meta: Dict[str, Dict[FullQual, Cannonical]] = {}
        self.examples = {}
        self.docs = {}
        self._doctree: Dict[str, str] = {}

    def get_example_data(
        self, example_section, *, obj: Any, qa: str, config: Config, log: logging.Logger
    ) -> Tuple[Section, List[Any]]:
        """Extract example section data from a NumpyDocString

        One of the section in numpydoc is "examples" that usually consist of number
        of paragraphs, interleaved with examples starting with >>> and ...,

        This attempts to parse this into structured data, with text, input and output
        as well as to infer the types of each token in the input examples.

        This is currently relatively limited as the inference does not work across
        code blocks.

        Parameters
        ----------
        example_section
            The example section of a numpydoc parsed docstring
        obj
            The current object. It is common for the current object/function to not
            have to be imported imported in docstrings. This should become a high
            level option at some point. Note that for method classes, the class should
            be made available but currently is not.
        qa : str
            The fully qualified name of current object
        config : Config
            Current configuration
        log
            Logger instance

        Examples
        --------
        Those are self examples, generating papyri documentation with papyri should
        be able to handle the following

        A simple input, should be execute and output should be shown if --exec option is set

        >>> 1+1

        >>> 2+2
        4

        Output with Syntax error should be marked as so.

        >>> [this is syntax error]

        if matplotlib and numpy available, we shoudl show graph

        >>> import matplotlib.pyplot as plt
        ... import numpy as np
        ... x = np.arange(0, 10, 0.1)
        ... plt.plot(x, np.sin(x))
        ... plt.show()

        Note that in the above we use `plt.show`,
        but we can configure papyri to automatically detect
        when figures are created.

        Notes
        -----
        We do not yet properly handle explicit exceptions in examples, and those are
        seen as Papyri failures.

        The capturing of matplotlib figures is also limited.
        """
        assert qa is not None
        example_code = "\n".join(example_section)
        import matplotlib.pyplot as plt

        if qa in config.exclude_jedi:
            config = config.replace(infer=False)
            log.debug(f"Turning off type inference for func {qa!r}")

        sys_stdout = sys.stdout

        def dbg(*args):
            for arg in args:
                sys_stdout.write(f"{arg}\n")
            sys_stdout.flush()

        try:
            filename = inspect.getfile(obj)
        except TypeError:
            filename = None
        try:
            lineno = inspect.getsourcelines(obj)[1]
        except (TypeError, OSError):
            lineno = None

        doctest_runner = PapyriDocTestRunner(
            gen=self,
            obj=obj,
            qa=qa,
            config=config,
            # TODO: Make optionflags configurable
            optionflags=doctest.ELLIPSIS,
        )
        example_section_data = Section([], None)

        def debugprint(*args):
            """
            version of print that capture current stdout to use during testing to debug
            """
            sys_stdout.write(" ".join(str(x) for x in args) + "\n")

        blocks = doctest.DocTestParser().parse(example_code, name=qa)
        for block in blocks:
            if isinstance(block, doctest.Example):
                doctests = doctest.DocTest(
                    [block],
                    globs=doctest_runner.globs,
                    name=qa,
                    filename=filename,
                    lineno=lineno,
                    docstring=example_code,
                )
                if config.execute_doctests:
                    doctest_runner.run(doctests, out=debugprint, clear_globs=False)
                    doctest_runner.globs.update(doctests.globs)
                    example_section_data.extend(
                        doctest_runner.get_example_section_data()
                    )
                else:
                    example_section_data.append(MText(block.source))
            elif block:
                example_section_data.append(MText(block))

        example_section_data = doctest_runner._compact(example_section_data)

        # TODO fix this if plt.close not called and still a lingering figure.
        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
        if len(fig_managers) != 0:
            print_(f"Unclosed figures in {qa}!!")
            plt.close("all")

        return processed_example_data(example_section_data), doctest_runner.figs

    def clean(self, where: Path):
        """
        Erase a doc bundle folder.
        """
        for _, path in progress(
            (where / "module").glob("*.json"),
            description="cleaning previous bundle 1/3",
        ):
            path.unlink()
        for _, path in progress(
            (where / "assets").glob("*"), description="cleaning previous bundle 2/3"
        ):
            path.unlink()
        for _, path in progress(
            (where / "docs").glob("*"), description="cleaning previous bundle 3/3"
        ):
            path.unlink()

        if (where / "module").exists():
            (where / "module").rmdir()
        if (where / "assets").exists():
            (where / "assets").rmdir()
        if (where / "papyri.json").exists():
            (where / "papyri.json").unlink()
        if (where / "docs").exists():
            (where / "docs").rmdir()

    def collect_narrative_docs(self):
        """
        Crawl the filesystem for all docs/rst files

        """
        if not self.config.docs_path:
            return
        path = Path(self.config.docs_path).expanduser()
        self.log.info("Scraping Documentation")
        files = list(path.glob("**/*.rst"))
        trees = {}
        title_map = {}
        blbs = {}
        with self.progress() as p2:
            task = p2.add_task("Parsing narrative", total=len(files))

            for p in files:
                p2.update(task, description=compress_user(str(p)).ljust(7))
                p2.advance(task)

                if any([k in str(p) for k in self.config.narrative_exclude]):
                    print_(f"Skipping {p} – excluded in config file")
                    continue

                assert p.is_file()
                parts = p.relative_to(path).parts
                assert parts[-1].endswith("rst")
                try:
                    data = ts.parse(p.read_bytes(), p)
                except Exception as e:
                    raise type(e)(f"{p=}")
                blob = DocBlob.new()
                key = ":".join(parts)[:-4]
                try:
                    dv = DVR(
                        key,
                        set(),
                        local_refs=set(),
                        substitution_defs={},
                        aliases={},
                        version=self._meta["version"],
                        config=self.config.directives,
                    )
                    blob.arbitrary = [dv.visit(s) for s in data]
                except Exception as e:
                    e.add_note(f"Error in {p!r}")
                    raise
                # if dv._tocs:
                trees[key] = dv._tocs

                blob.item_file = None
                blob.item_line = None
                blob.item_type = None
                blob.aliases = []
                blob.example_section_data = Section([], None)
                blob.see_also = []
                blob.signature = None
                blob.validate()
                titles = [s.title for s in blob.arbitrary if s.title]
                if not titles:
                    title = f"<No Title {key}>"
                else:
                    title = titles[0]
                title_map[key] = title
                if "generated" not in key and title_map[key] is None:
                    print_(key, title)

                blbs[key] = blob
        for k, b in blbs.items():
            self.docs[k] = b.to_json()

        self._doctree = {"tree": make_tree(trees), "titles": title_map}

    def write_narrative(self, where: Path) -> None:
        (where / "toc.json").write_text(json.dumps(self._doctree, indent=2))
        (where / "docs").mkdir(exist_ok=True)
        for file, v in self.docs.items():
            subf = where / "docs"
            subf.mkdir(exist_ok=True, parents=True)
            (subf / file).write_bytes(v)

    def write_examples(self, where: Path) -> None:
        (where / "examples").mkdir(exist_ok=True)
        for k, v in self.examples.items():
            (where / "examples" / k).write_bytes(v)

    def write_api(self, where: Path):
        """
        Write the API section of the DocBundle.
        """
        (where / "module").mkdir(exist_ok=True)
        for k, v in self.data.items():
            (where / "module" / (k + ".json")).write_bytes(v.to_json())

    def partial_write(self, where):
        self.write_api(where)

    def write(self, where: Path):
        """
        Write a DocBundle folder.
        """
        self.write_api(where)
        self.write_narrative(where)
        self.write_examples(where)
        self.write_assets(where)
        with (where / "papyri.json").open("w") as f:
            assert "version" in self._meta
            f.write(json.dumps(self._meta, indent=2, sort_keys=True))

    def write_assets(self, where: Path) -> None:
        assets = where / "assets"
        assets.mkdir()
        for k, v in self.bdata.items():
            (assets / k).write_bytes(v)

    def put(self, path: str, obj):
        """
        put some json data at the given path
        """
        self.data[path] = obj

    def put_raw(self, path: str, data: bytes):
        """
        put some rbinary data at the given path.
        """
        self.bdata[path] = data

    def _transform_1(self, blob: DocBlob, ndoc) -> DocBlob:
        """
        Populates DocBlob content field from numpydoc parsed docstring.

        """
        for k, v in ndoc._parsed_data.items():
            blob.content[k] = v
        for k, v in blob.content.items():
            assert isinstance(v, (str, list, dict)), type(v)
        return blob

    def _transform_2(self, blob: DocBlob, target_item, qa: str) -> DocBlob:
        """
        Try to find relative path WRT site package and populate item_file field
        for DocBlob.
        """
        # will not work for dev install. Maybe an option to set the root location ?
        item_file: Optional[str] = find_file(target_item)
        if item_file is not None and item_file.endswith("<string>"):
            # dynamically generated object (like dataclass __eq__ method
            item_file = None
        r = qa.split(".")[0]
        if item_file is not None:
            # TODO: find a better way to get a relative path with respect to the
            # root of the package ?
            for s in SITE_PACKAGE + [
                os.path.expanduser(f"~/dev/{r}/"),
                os.path.expanduser("~"),
                os.getcwd(),
            ]:
                if item_file.startswith(s):
                    item_file = item_file[len(s) :]
        blob.item_file = item_file
        if item_file is None:
            if type(target_item).__name__ in (
                "builtin_function_or_method",
                "fused_cython_function",
                "cython_function_or_method",
            ):
                self.log.debug(
                    "Could not find source file for built-in function method."
                    "Likely compiled extension %s %s %s, will not be able to link to it.",
                    repr(qa),
                    target_item,
                    repr(type(target_item).__name__),
                )
            else:
                rqan = str(qa).split(".")[-1]
                if not (rqan.startswith("__") and rqan.endswith("__")):
                    self.log.warning(
                        "Could not find source file for %s (%s) [%s], will not be able to link to it.",
                        repr(qa) + ":" + rqan,
                        target_item,
                        type(target_item).__name__,
                    )

        return blob

    def _transform_3(self, blob, target_item):
        """
        Try to find source line number for target object and populate item_line
        field for DocBlob.
        """
        item_line = None
        try:
            item_line = inspect.getsourcelines(target_item)[1]
        except OSError:
            self.log.debug("Could not find item_line for %s, (OSERROR)", target_item)
        except TypeError:
            if type(target_item).__name__ in (
                "builtin_function_or_method",
                "fused_cython_function",
                "cython_function_or_method",
            ):
                self.log.debug(
                    "Could not find item_line for %s, (TYPEERROR), likely from a .so file",
                    target_item,
                )
            else:
                self.log.debug(
                    "Could not find item_line for %s, (TYPEERROR)", target_item
                )
        blob.item_line = item_line

        return blob

    def prepare_doc_for_one_object(
        self,
        target_item: Any,
        ndoc,
        *,
        qa: str,
        config: Config,
        aliases: List[str],
        api_object: APIObjectInfo,
    ) -> Tuple[DocBlob, List]:
        """
        Get documentation information for one python object

        Parameters
        ----------
        target_item : any
            the object you want to get documentation for
        ndoc
            numpydoc parsed docstring.
        qa : str
            fully qualified object path.
        config : Config
            current configuratin
        aliases : sequence
            other aliases for cuttent object.
        api_object : APIObjectInfo
            Describes the object's type and other relevant information

        Returns
        -------
        Tuple of two items,
        blob:
            DocBlob with info for current object.
        figs:
            dict mapping figure names to figure data.

        See Also
        --------
        collect_api_docs
        """
        assert isinstance(aliases, list)
        blob: DocBlob = DocBlob.new()

        blob = self._transform_1(blob, ndoc)
        blob = self._transform_2(blob, target_item, qa)
        blob = self._transform_3(blob, target_item)
        assert set(blob.content.keys()) == set(blob.ordered_sections), (
            set(blob.content.keys()) - set(blob.ordered_sections),
            set(blob.ordered_sections) - set(blob.content.keys()),
        )

        item_type = str(type(target_item))
        if blob.content["Signature"]:
            try:
                # the type ignore below is wrong and need to be refactored.
                # we basically modify blob.content in place, but should not.
                if "Signature" in blob.content:
                    ss = blob.content["Signature"]
                    del blob.content["Signature"]
                else:
                    ss = None
                sig = ObjectSignature.from_str(ss)  # type: ignore
                if sig is not None:
                    blob.signature = sig.to_node()
            except TextSignatureParsingFailed:
                # this really fails often when the first line is not Signature.
                # or when numpy has the def f(,...[a,b,c]) optional parameter.
                pass
        else:
            assert blob is not None
            assert api_object is not None
            if api_object.signature is None:
                blob.signature = None
            else:
                blob.signature = api_object.signature.to_node()
            del blob.content["Signature"]
        self.log.debug("SIG %r", blob.signature)

        if api_object.special("Examples"):
            # warnings this is true only for non-modules
            # things.
            try:
                example_section_data, figs = self.get_example_data(
                    api_object.special("Examples").value,
                    obj=target_item,
                    qa=qa,
                    config=config,
                    log=self.log,
                )
            except Exception as e:
                example_section_data = Section([], None)
                self.log.error("Error getting example data in %s", repr(qa))
                from .errors import ExampleError1

                raise ExampleError1(f"Error getting example data in {qa!r}") from e
        else:
            example_section_data = Section([], None)
            figs = []

        refs_I = []
        refs_Ib = []
        if ndoc["See Also"]:
            for line in ndoc["See Also"]:
                rt, desc = line
                assert isinstance(desc, list), line
                for ref, _type in rt:
                    refs_I.append(ref)
        if api_object.special("See Also"):
            for sa in api_object.special("See Also").value:
                refs_Ib.append(sa.name.value)

        if api_object.kind != "module":
            # TODO: most module docstring are not properly parsed by numpydoc.
            # but some are.
            assert refs_I == refs_Ib, (refs_I, refs_Ib)

        blob.example_section_data = example_section_data

        blob.item_type = item_type

        del blob.content["Examples"]
        del blob.content["index"]

        ref = blob.content["References"]
        if ref == "":
            blob.references = None
        else:
            blob.references = ref
        del blob.content["References"]

        blob.aliases = aliases
        assert set(blob.content.keys()) == set(blob.ordered_sections), (
            set(blob.content.keys()),
            set(blob.ordered_sections),
        )
        for section in ["Extended Summary", "Summary", "Notes", "Warnings"]:
            try:
                data = blob.content.get(section, None)
                if data is None:
                    # don't exists
                    pass
                elif not data:
                    # is empty
                    blob.content[section] = Section([], None)
                else:
                    tsc = ts.parse("\n".join(data).encode(), qa)
                    assert len(tsc) in (0, 1), (tsc, data)
                    if tsc:
                        tssc = tsc[0]
                    else:
                        tssc = Section([], None)
                    assert isinstance(tssc, Section)
                    blob.content[section] = tssc
            except Exception:
                self.log.exception(f"Skipping section {section!r} in {qa!r} (Error)")
                raise
        assert isinstance(blob.content["Summary"], Section)
        assert isinstance(
            blob.content.get("Summary", Section([], None)), Section
        ), blob.content["Summary"]

        sections_ = [
            "Parameters",
            "Returns",
            "Raises",
            "Yields",
            "Attributes",
            "Other Parameters",
            "Warns",
            "Methods",
            "Receives",
        ]

        for s in set(sections_).intersection(blob.content.keys()):
            assert isinstance(blob.content[s], list), f"{s}, {blob.content[s]} {qa} "
            new_content = []

            for param, type_, desc in blob.content[s]:
                assert isinstance(desc, list)
                items = []
                if desc:
                    try:
                        items = parse_rst_section("\n".join(desc), qa)
                    except Exception as e:
                        raise type(e)(f"from {qa}")
                    for l in items:
                        assert not isinstance(l, Section)
                new_content.append(Param(param, type_, desc=items).validate())
            if new_content:
                blob.content[s] = Section([Parameters(new_content)], None)
            else:
                blob.content[s] = Section([], None)

        blob.see_also = _normalize_see_also(blob.content.get("See Also", Section()), qa)
        del blob.content["See Also"]

        assert set(blob.content.keys()) == set(blob.ordered_sections), (
            set(blob.content.keys()),
            set(blob.ordered_sections),
        )
        return blob, figs

    def collect_examples(self, folder: Path, config):
        acc = []
        examples = list(folder.glob("**/*.py"))

        valid_examples = []
        for e in examples:
            if any(str(e).endswith(p) for p in config.examples_exclude):
                continue
            valid_examples.append(e)
        examples = valid_examples

        # TODO: resolve this path with respect the configuration file.
        # this is of course if we have configuration file.
        #        assert (
        #            len(examples) > 0
        #        ), "we havent' found any examples, it is likely that the path is incorrect."

        with self.progress() as p2:
            failed = []

            taskp = p2.add_task(description="Collecting examples", total=len(examples))
            for example in examples:
                p2.update(taskp, description=compress_user(str(example)).ljust(7))
                p2.advance(taskp)
                executor = BlockExecutor({})
                script = example.read_text()
                ce_status = "None"
                figs = []
                if config.execute_doctests:
                    with executor:
                        try:
                            executor.exec(script, name=str(example))
                            figs = [
                                (f"ex-{example.name}-{i}.png", f)
                                for i, f in enumerate(executor.get_figs())
                            ]
                            ce_status = "execed"
                        except Exception as e:
                            failed.append(str(example))
                            if config.exec_failure == "fallback":
                                self.log.exception("%s failed %s", example, type(e))
                            else:
                                raise type(e)(f"Within {example}")
                entries_p = parse_script(
                    script,
                    ns={},
                    prev="",
                    config=config,
                )

                entries: List[Any]
                if entries_p is None:
                    print_("Issue in ", example)
                    entries = [("fail", "fail")]
                else:
                    entries = list(entries_p)

                assert isinstance(entries, list), entries

                entries = _add_classes(entries)
                assert set(len(x) for x in entries) == {3}

                tok_entries = [GenToken(*x) for x in entries]
                l: List[Any] = []  # get typechecker to shut up.
                s = Section(
                    l
                    + [Code(tok_entries, "", ce_status)]  # ignore: type
                    + [
                        Fig(
                            RefInfo.from_untrusted(
                                self.root, self.version, "assets", name
                            )
                        )  # ignore: type
                        for name, _ in figs
                    ],  # ignore: type
                    None,
                )
                s = processed_example_data(s)
                dv = DVR(
                    example.name,
                    frozenset(),
                    local_refs=frozenset(),
                    substitution_defs={},
                    aliases={},
                    version=self.version,
                    config=self.config.directives,
                )
                s2 = dv.visit(s)

                acc.append(
                    (
                        {example.name: s2},
                        figs,
                    )
                )
        assert len(failed) == 0, failed
        return acc

    def _get_collector(self) -> DFSCollector:
        """
        Construct a depth first search collector that will try to find all
        the objects it can.

        We give it the root module, and a few submodules as seed.
        """
        assert "." not in self.root
        n0 = __import__(self.root)
        submodules = []

        subs = self.config.submodules
        extra_from_conf = [self.root + "." + s for s in subs]
        for name in extra_from_conf:
            _, *r = name.split(".")
            nx = __import__(name)
            for sub in r:
                nx = getattr(nx, sub)
            submodules.append(nx)

        self.log.debug(
            "Collecting API starting from [%r], and %s",
            n0.__name__,
            [m.__name__ for m in submodules],
        )
        return DFSCollector(n0, submodules)

    def collect_examples_out(self):
        examples_folder = self.config.examples_folder
        self.log.debug("Example Folder: %s", examples_folder)
        if examples_folder is not None:
            examples_folder = Path(examples_folder).expanduser()
            examples_data = self.collect_examples(
                examples_folder,
                config=self.config,
            )
            for edoc, figs in examples_data:
                self.examples.update({k: v.to_json() for k, v in edoc.items()})
                for name, data in figs:
                    self.put_raw(name, data)

    def extract_docstring(
        self, *, qa: str, target_item: Any
    ) -> Tuple[Optional[str], List[Section], APIObjectInfo]:
        """
        Extract docstring from an object.

        Detects whether an object includes a docstring and parses the object's
        type.

        Parameters
        ----------
        qa : str
            Fully qualified name of the object we are extracting the
            documentation from
        target_item : Any
            Object we wish inspect. Can be any kind of object.

        Returns
        -------
        item_docstring : str
            The unprocessed object's docstring
        sections : list of Section
            A list of serialized sections of the docstring
        api_object : APIObjectInfo
            Describes the object's type and other relevant information

        """
        item_docstring: str = target_item.__doc__
        if item_docstring is not None:
            item_docstring = dedent_but_first(item_docstring)
        builtin_function_or_method = type(sum)

        if isinstance(target_item, ModuleType):
            api_object = APIObjectInfo(
                "module", item_docstring, None, target_item.__name__, qa
            )
        elif isinstance(
            target_item, (FunctionType, builtin_function_or_method)
        ) or callable(target_item):
            sig: Optional[ObjectSignature]
            try:
                sig = ObjectSignature(target_item)
            except (ValueError, TypeError):
                sig = None
            try:
                api_object = APIObjectInfo(
                    "function", item_docstring, sig, target_item.__name__, qa
                )
            except Exception as e:
                e.add_note(f"For object {qa!r}")
                raise
        elif isinstance(target_item, type):
            api_object = APIObjectInfo(
                "class", item_docstring, None, target_item.__name__, qa
            )
        else:
            api_object = APIObjectInfo(
                "other", item_docstring, None, target_item.__name__, qa
            )
            # print_("Other", target_item)
            # assert False, type(target_item)

        if item_docstring is None and not isinstance(target_item, ModuleType):
            return None, [], api_object
        elif item_docstring is None and isinstance(target_item, ModuleType):
            item_docstring = """This module has no documentation"""

        try:
            sections = ts.parse(item_docstring.encode(), qa)
        except (AssertionError, NotImplementedError) as e:
            self.log.error("TS could not parse %s, %s", repr(qa), e)
            raise type(e)(f"from {qa}") from e
            sections = []
        except Exception as e:
            raise type(e)(f"from {qa}")

        assert api_object is not None
        return item_docstring, sections, api_object

    def collect_package_metadata(self, root, relative_dir, meta):
        """
        Try to gather generic metadata about the current package we are going to
        build the documentation for.
        """
        self.root = root
        if self.config.logo:
            logo_path = relative_dir / self.config.logo
            self.put_raw(logo_path.name, logo_path.read_bytes())
            logo = logo_path.name
        else:
            logo = None
        module = __import__(root)
        # TODO: xarray does not have __version__ anymore, find another logic
        self.version = getattr(module, "__version__", "0.0.0")
        assert parse(self.version)

        try:
            meta["tag"] = meta["tag"].format(version=self.version)
        except KeyError:
            meta["tag"] = self.version

        self._meta.update({"logo": logo, "module": root, "version": self.version})
        self._meta.update(meta)

    def collect_api_docs(self, root: str, limit_to: List[str]) -> None:
        """
        Crawl one module and stores resulting DocBundle in json files.

        Parameters
        ----------
        root : str
            Module name to generate DocBundle for.
        limit_to : list of string
            For partial documentation building and testing purposes
            we may want to generate documentation for only a single item.
            If this list is non-empty we will collect documentation
            just for these items.

        See Also
        --------
        prepare_doc_for_one_object

        """

        collector: DFSCollector = self._get_collector()
        collected: Dict[str, Any] = collector.items()

        # collect all items we want to document.
        excluded = sorted(self.config.exclude)
        if excluded:
            self.log.info(
                "The following items will be excluded by the configurations:\n %s",
                json.dumps(excluded, indent=2, sort_keys=True),
            )
        else:
            self.log.info("No items excluded by the configuration")
        missing = list(set(excluded) - set(collected.keys()))
        if missing:
            self.log.warning(
                "The following items have been excluded but were not found:\n %s",
                json.dumps(sorted(missing), indent=2, sort_keys=True),
            )

        collected = {k: v for k, v in collected.items() if k not in excluded}

        if limit_to:
            non_existinsing = [k for k in limit_to if k not in collected]
            if non_existinsing:
                self.log.warning(
                    "You asked to build docs only for following items,"
                    " but they don't exist:\n %s, existing items are %s",
                    non_existinsing,
                    collected.keys(),
                )
            collected = {k: v for k, v in collected.items() if k in limit_to}
            self.log.info("DEV: regenerating docs only for")
            for k, v in collected.items():
                self.log.info(f"    {k}:{v}")

        aliases: Dict[FullQual, Cannonical]
        aliases, not_found = collector.compute_aliases()
        rev_aliases: Dict[Cannonical, FullQual] = {v: k for k, v in aliases.items()}

        known_refs = frozenset(
            {
                RefInfo.from_untrusted(root, self.version, "module", qa)
                for qa in collected.keys()
            }
        )

        error_collector = ErrorCollector(self.config, self.log)
        # with self.progress() as p2:
        # just nice display of progression.
        # taskp = p2.add_task(description="parsing", total=len(collected))

        failure_collection: Dict[str, List[str]] = defaultdict(lambda: [])
        api_object: APIObjectInfo
        for qa, target_item in collected.items():
            self.log.debug("treating %r", qa)

            with error_collector(qa=qa) as ecollector:
                item_docstring, arbitrary, api_object = self.extract_docstring(
                    qa=qa,
                    target_item=target_item,
                )
                self.log.debug("APIOBJECT %r", api_object)
            if ecollector.errored:
                if ecollector._unexpected_errors.keys():
                    self.log.warning(
                        "error with %s %s",
                        qa,
                        list(ecollector._unexpected_errors.keys()),
                    )
                else:
                    self.log.info(
                        "only expected error with %s, %s",
                        qa,
                        list(ecollector._expected_errors.keys()),
                    )
                continue

            try:
                if item_docstring is None:
                    ndoc = NumpyDocString(dedent_but_first("No Docstrings"))
                else:
                    ndoc = NumpyDocString(dedent_but_first(item_docstring))
                    # note currently in ndoc we use:
                    # _parsed_data
                    # direct access to  ["See Also"], and [""]
                    # and :
                    # ndoc.ordered_sections
            except Exception as e:
                if not isinstance(target_item, ModuleType):
                    self.log.exception(
                        "Unexpected error parsing %s – %s",
                        qa,
                        target_item.__name__,
                    )
                    failure_collection["NumpydocError-" + str(type(e))].append(qa)
                if isinstance(target_item, ModuleType):
                    # TODO: ndoc-placeholder : remove placeholder here
                    ndoc = NumpyDocString(f"To remove in the future –– {qa}")
                else:
                    continue
            if not isinstance(target_item, ModuleType):
                arbitrary = []
            ex = self.config.execute_doctests
            if self.config.execute_doctests and any(
                qa.startswith(pat) for pat in self.config.execute_exclude_patterns
            ):
                ex = False

            # TODO: ndoc-placeholder : make sure ndoc placeholder handled here.
            with error_collector(qa=qa) as c:
                doc_blob, figs = self.prepare_doc_for_one_object(
                    target_item,
                    ndoc,
                    qa=qa,
                    config=self.config.replace(execute_doctests=ex),
                    aliases=collector.aliases[qa],
                    api_object=api_object,
                )
            del api_object
            if c.errored:
                continue
            _local_refs: List[str] = []

            sections_ = [
                "Parameters",
                "Returns",
                "Raises",
                "Yields",
                "Attributes",
                "Other Parameters",
                "Warns",
                ##"Warnings",
                "Methods",
                # "Summary",
                "Receives",
            ]
            for s in sections_:
                for child in doc_blob.content.get(s, []):
                    if isinstance(child, Parameters):
                        for param in child.children:
                            new_ref = [u.strip() for u in param[0].split(",") if u]
                            if new_ref:
                                _local_refs = _local_refs + new_ref

            # substitution_defs: Dict[str, Union(MImage, ReplaceNode)] = {}
            substitution_defs = {}
            for section in doc_blob.sections:
                for child in doc_blob.content.get(section, []):
                    if isinstance(child, SubstitutionDef):
                        substitution_defs[child.value] = child.children

            # def flat(l) -> List[str]:
            #    return [y for x in l for y in x]
            for lr1 in _local_refs:
                assert isinstance(lr1, str)
            # lr: FrozenSet[str] = frozenset(flat(_local_refs))
            lr: FrozenSet[str] = frozenset(_local_refs)

            dv = DVR(
                qa,
                known_refs,
                local_refs=lr,
                substitution_defs=substitution_defs,
                aliases={},
                version=self.version,
                config=self.config.directives,
            )

            doc_blob.arbitrary = [dv.visit(s) for s in arbitrary]
            doc_blob.example_section_data = dv.visit(doc_blob.example_section_data)
            doc_blob._content = {k: dv.visit(v) for (k, v) in doc_blob._content.items()}

            for section in ["Extended Summary", "Summary", "Notes"] + sections_:
                if section in doc_blob.content:
                    doc_blob.content[section] = dv.visit(doc_blob.content[section])

            for sa in doc_blob.see_also:
                from .tree import resolve_

                r = resolve_(
                    qa,
                    known_refs,
                    frozenset(),
                    sa.name.value,
                    rev_aliases=rev_aliases,
                )
                assert isinstance(r, RefInfo)
                if r.kind == "module":
                    sa.name.reference = r
                else:
                    imp = DVR._import_solver(sa.name.value)
                    if imp:
                        self.log.debug(
                            "TODO: see also resolve for %s in %s, %s",
                            sa.name.value,
                            qa,
                            imp,
                        )

            # end processing
            assert not isinstance(doc_blob._content, str), doc_blob._content
            try:
                doc_blob.validate()
            except Exception as e:
                e.add_note(f"Error in {qa}")
                raise
            self.log.debug(doc_blob.signature)
            self.put(qa, doc_blob)
            if figs:
                self.log.debug("Found %s figures", len(figs))
            for name, data in figs:
                self.put_raw(name, data)
        if error_collector._unexpected_errors:
            self.log.info(
                "ERRORS:"
                + tomli_w.dumps(error_collector._unexpected_errors).replace(
                    ",", ",    \n"
                )
            )
        if error_collector._expected_unseen:
            inverted = defaultdict(lambda: [])
            for qa, errs in error_collector._expected_unseen.items():
                for err in errs:
                    inverted[err].append(qa)
            self.log.info("UNSEEN ERRORS:" + tomli_w.dumps(inverted))
        if failure_collection:
            self.log.info(
                "The following parsing failed \n%s",
                json.dumps(failure_collection, indent=2, sort_keys=True),
            )
        self._meta.update(
            {
                "aliases": aliases,
            }
        )


def is_private(path):
    """
    Determine if a import path, or fully qualified is private.
    that usually implies that (one of) the path part starts with a single underscore.
    """
    return any(p.startswith("_") and not p.startswith("__") for p in path.split("."))


def find_cannonical(qa: str, aliases: List[str]):
    """
    Given the fully qualified name and a lit of aliases, try to find the canonical one.

    The canonical name is usually:
        - short (less depth in number of modules)
        - does not contain special chars like <, > for locals
        - none of the part start with _.
        - if there are many names that have the same depth and are shorted than the qa, we bail.

    We might want to be careful with dunders.

    If we can't find a canonical, there are many, or are identical to the fqa, return None.
    """

    def _level(c):
        return c.count(".") + c.count(":")

    qa_level = _level(qa)
    min_alias_level = min(_level(a) for a in set(aliases))
    if min_alias_level < qa_level:
        shorter_candidates = [c for c in aliases if _level(c) <= min_alias_level]
    else:
        shorter_candidates = [c for c in aliases if _level(c) <= qa_level]
    if (
        len(shorter_candidates) == 1
        and not is_private(shorter_candidates[0])
        and shorter_candidates[0] != qa
    ):
        return shorter_candidates[0]
    return None
