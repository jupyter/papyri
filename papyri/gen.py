"""
Main module responsible from scrapping the code, docstrings, an (TBD) rst files,
and turning that into intermediate representation files that can be published.

This also does some code execution, and inlining of figures, though that should
likely be separated into a separate module at some point.


"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import re
import site
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from itertools import count
from types import FunctionType, ModuleType
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import jedi
import toml
from IPython.core.oinspect import find_file
from pygments import lex
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn
from there import print
from velin.examples_section_utils import InOut, splitblank, splitcode

from .miscs import BlockExecutor, DummyP
from .take2 import (
    Code,
    Fig,
    Node,
    Param,
    Ref,
    RefInfo,
    Section,
    SeeAlsoItem,
    Text,
    parse_rst_section,
)
from .tree import DirectiveVisiter
from .utils import TimeElapsedColumn, dedent_but_first, pos_to_nl, progress
from .vref import NumpyDocString

try:
    from . import ts
except (ImportError, OSError):
    import sys

    sys.exit(
        """
            Tree Sitter RST parser not available, you may need to:

            $ git clone https://github.com/stsewd/tree-sitter-rst
            $ papyri build-parser
            """
    )
SITE_PACKAGE = site.getsitepackages()


def paragraph(lines: List[str]) -> Any:
    """
    Leftover rst parsing,

    Remove at some point.
    """
    [section] = ts.parse("\n".join(lines).encode())
    assert len(section.children) == 1
    p2 = section.children[0]
    return p2


_JEDI_CACHE = Path("~/.cache/papyri/jedi/").expanduser()


from hashlib import sha256
import datetime

from contextlib import contextmanager


@contextmanager
def with_context(**kwargs):
    try:
        yield
    except Exception as e:
        raise type(e)(f"With context {kwargs}") from e


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

    Return
    ------
    List of tuples with:

    index: int
        index in the tokenstream
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
                    print(
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
    return acc


from enum import Enum


class ExecutionStatus(Enum):
    none = "None"
    compiled = "compiled"
    syntax_error = "syntax_error"
    exec_error = "exception_in_exec"


def _execute_inout(item):
    script = "\n".join(item.in_)
    ce_status = ExecutionStatus.none
    try:
        compile(script, "<>", "exec")
        ce_status = ExecutionStatus.compiled
    except SyntaxError:
        ce_status = ExecutionStatus.syntax_error

    return script, item.out, ce_status.value


def get_example_data(
    example_section, *, obj, qa: str, config, log
) -> Tuple[Section, List[Any]]:
    """Extract example section data from a NumpyDocstring

    One of the section in numpydoc is "examples" that usually consist of number
    if paragraph, interleaved with examples starting with >>> and ...,

    This attempt to parse this into structured data, with text, input and output
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
    qa
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
    blocks = list(map(splitcode, splitblank(example_section)))
    example_section_data = Section()
    import matplotlib.pyplot as plt
    import numpy as np

    acc = ""
    figure_names = (f"fig-{qa}-{i}.png" for i in count(0))
    ns = {"np": np, "plt": plt, obj.__name__: obj}
    executor = BlockExecutor(ns)
    figs = []
    # fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
    fig_managers = executor.fig_man()
    assert (len(fig_managers)) == 0, f"init fail in {qa} {len(fig_managers)}"
    wait_for_show = config.wait_for_plt_show
    if qa in config.exclude_jedi:
        config = config.replace(infer=False)
        log.debug(f"Turning off type inference for func {qa!r}")
    chunks = (it for block in blocks for it in block)
    with executor:
        for item in chunks:
            if isinstance(item, InOut):
                script, out, ce_status = _execute_inout(item)
                figname = None
                raise_in_fig = None
                did_except = False
                if config.exec and ce_status == ExecutionStatus.compiled.value:
                    if not wait_for_show:
                        # we should aways have 0 figures
                        # unless stated otherwise
                        assert len(fig_managers) == 0
                    try:
                        res = object()
                        try:
                            res, fig_managers, sout, serr = executor.exec(script)
                            ce_status = "execed"
                        except Exception:
                            log.exception("error in execution: %s", qa)
                            ce_status = "exception_in_exec"
                            if config.exec_failure != "fallback":
                                raise
                        if fig_managers and (
                            ("plt.show" in script) or not wait_for_show
                        ):
                            raise_in_fig = True
                            for fig, figname in zip(executor.get_figs(), figure_names):
                                figs.append((figname, fig))
                            plt.close("all")
                            raise_in_fig = False

                    except Exception:
                        did_except = True
                        print(f"exception executing... {qa}")
                        fig_managers = executor.fig_man()
                        if raise_in_fig or config.exec_failure != "fallback":
                            raise
                    finally:
                        if not wait_for_show:
                            if fig_managers:
                                for fig, figname in zip(
                                    executor.get_figs(), figure_names
                                ):
                                    figs.append((figname, fig))
                                    print(
                                        f"Still fig manager(s) open for {qa}: {figname}"
                                    )
                                plt.close("all")
                            fig_managers = executor.fig_man()
                            assert len(fig_managers) == 0, fig_managers + [
                                did_except,
                            ]
                    # we've executed, we now want to compare output
                    # in the docstring with the one we produced.
                    if (out == repr(res)) or (res is None and out == []):
                        pass
                    else:
                        pass
                        # captured output differ TBD
                entries = parse_script(script, ns=ns, prev=acc, config=config, where=qa)
                if entries is None:
                    entries = [("jedi failed", "jedi failed")]

                acc += "\n" + script
                example_section_data.append(
                    Code(entries, "\n".join(item.out), ce_status)
                )
                if figname:
                    example_section_data.append(Fig(figname))
            else:
                assert isinstance(item.out, list)
                example_section_data.append(Text("\n".join(item.out)))

    # TODO fix this if plt.close not called and still a ligering figure.
    fig_managers = executor.fig_man()
    if len(fig_managers) != 0:
        print(f"Unclosed figures in {qa}!!")
        plt.close("all")

    return processed_example_data(example_section_data), figs


def get_classes(code):
    """
    Extract Pygments token classes names for given code block
    """
    list(lex(code, PythonLexer()))
    FMT = HtmlFormatter()
    classes = [FMT.ttype2class.get(x) for x, y in lex(code, PythonLexer())]
    classes = [c if c is not None else "" for c in classes]
    return classes


def processed_example_data(example_section_data) -> Section:
    """this should be no-op on already ingested"""
    new_example_section_data = Section()
    for in_out in example_section_data:
        type_ = in_out.__class__.__name__
        # color examples with pygments classes
        if type_ == "Text":
            blocks = parse_rst_section(in_out.value)
            for b in blocks:
                new_example_section_data.append(b)

        elif type_ == "Code":
            in_ = in_out.entries
            # assert len(in_[0]) == 3, len(in_[0])
            if len(in_[0]) == 2:
                text = "".join([x for x, y in in_])
                classes = get_classes(text)
                in_out.entries = [ii + (cc,) for ii, cc in zip(in_, classes)]
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
    exec: bool = False
    source: Optional[str] = None
    homepage: Optional[str] = None
    docs: Optional[str] = None
    docs_path: Optional[str] = None
    wait_for_plt_show: Optional[bool] = True
    examples_exclude: Sequence[str] = ()
    exclude_jedi: Sequence[str] = ()

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def load_configuration(path: str) -> Tuple[str, MutableMapping[str, Any]]:
    """
    Given a path, load a configuration from a File.
    """
    conffile = Path(path).expanduser()
    if conffile.exists():
        conf: MutableMapping[str, Any] = toml.loads(conffile.read_text())
        assert len(conf.keys()) == 1
        root = next(iter(conf.keys()))
        return root, conf[root]
    else:
        sys.exit(f"{conffile!r} does not exists.")


def gen_main(
    infer: Optional[bool],
    exec_: Optional[bool],
    target_file: str,
    debug,
    *,
    dummy_progress: bool,
    dry_run=bool,
    api: bool,
    examples: bool,
    fail,
    narrative,
) -> None:
    """
    Main entry point to generate docbundle files,

    This will take care of reading  single configuration file with the option
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

    Returns
    -------
    None

    """
    target_module_name, conf = load_configuration(target_file)
    config = Config(**conf, dry_run=dry_run, dummy_progress=dummy_progress)
    if exec_ is not None:
        config.exec = exec_
    if infer is not None:
        config.infer = infer

    target_dir = Path("~/.papyri/data").expanduser()

    if not target_dir.exists() and not config.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    g = Gen(dummy_progress=dummy_progress, config=config)
    g.log.info("Will write data to %s", target_dir)
    if debug:
        g.log.setLevel("DEBUG")
        g.log.debug("Log level set to debug")

    g.collect_package_metadata(
        target_module_name,
        relative_dir=Path(target_file).parent,
    )
    if examples:
        g.collect_examples_out()
    if api:
        g.collect_api_docs(target_module_name)
    if narrative:
        g.collect_narrative_docs()
    if not config.dry_run:
        p = target_dir / (g.root + "_" + g.version)
        p.mkdir(exist_ok=True)

        g.log.info("Saving current Doc bundle to %s", p)
        g.clean(p)
        g.write(p)


def full_qual(obj):
    if isinstance(obj, ModuleType):
        return obj.__name__
    else:
        try:
            if hasattr(obj, "__qualname__") and (
                getattr(obj, "__module__", None) is not None
            ):
                return obj.__module__ + "." + obj.__qualname__
            elif hasattr(obj, "__name__") and (
                getattr(obj, "__module__", None) is not None
            ):
                return obj.__module__ + "." + obj.__name__
        except Exception:
            pass
        return None
    return None


class DFSCollector:
    """
    Depth first search collector.

    Will scan documentation to find all reachable items in the namespace
    of our root object (we don't want to go scan other libraries).

    Three was some issues with BFS collector originally, I'm not sure I remember what.


    """

    def __init__(self, root, others):
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
        self.aliases = defaultdict(lambda: [])
        self._open_list = [(root, [root.__name__])]
        for o in others:
            self._open_list.append((o, o.__name__.split(".")))

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
                print("after import qa differs : {qa} -> {nqa}")
                if self.obj[nqa] == item:
                    print("present twice")
                    del self.obj[nqa]
                else:
                    print("differs: {item} != {other}")

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
        if not qa.split(".")[0] == self.root:
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
            self._open_list.append((getattr(mod, k), stack + [k]))

    def visit_ClassType(self, klass, stack):
        for k, v in klass.__dict__.items():
            self._open_list.append((v, stack + [k]))

    def visit_FunctionType(self, fun, stack):
        pass


class DocBlob(Node):
    """
    An object containing information about the documentation of an arbitrary object.

    Instead of docblob begin a NumpyDocString, I'm thinking of them having a numpydocstring.
    This helps with arbitraty documents (module, examples files) that cannot be parsed by Numpydoc,
    as well as link to external references, like images generated.
    """

    @classmethod
    def _deserialise(cls, **kwargs):
        # print("will deserialise", cls)
        try:
            instance = cls._instance()
        except Exception as e:
            raise type(e)(f"Error deserialising {cls}, {kwargs})") from e
        assert "_content" in kwargs
        assert kwargs["_content"] is not None
        for k, v in kwargs.items():
            setattr(instance, k, v)
        return instance

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
        "index",
    ]  # List of sections in order

    _content: Dict[str, Optional[Section]]
    refs: List[str]
    ordered_sections: List[str]
    item_file: Optional[str]
    item_line: Optional[int]
    item_type: Optional[str]
    aliases: List[str]
    example_section_data: Section
    see_also: List[SeeAlsoItem]  # see also data
    signature: Optional[str]
    references: Optional[List[str]]
    arbitrary: List[Section]

    __slots__ = (
        "_content",
        "example_section_data",
        "refs",
        "ordered_sections",
        "signature",
        "item_file",
        "item_line",
        "item_type",
        "aliases",
        "see_also",
        "references",
        "logo",
        "arbitrary",
    )

    def __repr__(self):
        return "<DocBlob ...>"

    def slots(self):
        return [
            "_content",
            "example_section_data",
            "refs",
            "ordered_sections",
            "item_file",
            "item_line",
            "item_type",
            "signature",
            "references",
            "aliases",
            "arbitrary",
        ]

    def __init__(self):
        self._content = None
        self.example_section_data = None
        self.refs = None
        self.ordered_sections = None
        self.item_file = None
        self.item_line = None
        self.item_type = None
        self.aliases = []
        self.signature = None

    @property
    def content(self):
        """
        List of sections in the doc blob docstrings

        """
        return self._content

    @content.setter
    def content(self, new):
        assert not new.keys() - {
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
            "index",
        }
        self._content = new
        assert self._content is not None


def _numpy_data_to_section(data: List[Tuple[str, str, List[str]]], title: str):
    assert isinstance(data, list), repr(data)
    acc = []
    for param, type_, desc in data:
        assert isinstance(desc, list)
        items = []
        if desc:
            items = parse_rst_section("\n".join(desc))
            for l in items:
                assert not isinstance(l, Section)
        acc.append(Param(param, type_, desc=items).validate())
    return Section(acc, title)


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
_special = {"See Also", "Examples", "Signature"}

from .take2 import NumpydocExample, NumpydocSeeAlso, NumpydocSignature


class APIObjectInfo:
    """
    Info about an API object
    This object can be many things:

    Module, Class, method, function.

    I'll see how I handle that later.
    """

    kind: str
    docstring: str

    def __init__(self, kind, docstring):
        self.kind = kind
        self.docstring = docstring
        self.parsed = []
        if docstring is not None and kind != "module":
            # TS is going to choke on this as See Also and other
            # sections are technically invalid.
            ndoc = NumpyDocString(dedent_but_first(docstring))

            for title in ndoc.ordered_sections:
                if not ndoc[title]:
                    continue
                if title in _numpydoc_sections_with_param:
                    section = _numpy_data_to_section(ndoc[title], title)
                    assert isinstance(section, Section)
                    self.parsed.append(section)
                elif title in _numpydoc_sections_with_text:
                    docs = ts.parse("\n".join(ndoc[title]).encode())
                    assert len(docs) == 1, ("\n".join(ndoc[title]), docs)
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
            self.parsed = ts.parse(docstring.encode())
        self.validate()

    def special(self, title):
        if self.kind == "module":
            return []
        res = [s for s in self.parsed if s.title == title]
        if not res:
            return []
        assert len(res) == 1
        assert not isinstance(res[0], Section), self.parsed
        return res[0]

    def validate(self):
        for p in self.parsed:
            assert isinstance(
                p, (Section, NumpydocExample, NumpydocSeeAlso, NumpydocSignature)
            )
            p.validate()


def _normalize_see_also(see_also: List[Any], qa):
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
    section: Section
    name_and_types: List[Tuple[str, str]]
    name: str
    type_or_description: str

    for name_and_types, raw_description in see_also:
        try:
            for (name, type_or_description) in name_and_types:
                if type_or_description and not raw_description:
                    assert isinstance(type_or_description, str)
                    type_ = None
                    # we have all in a single line,
                    # and there is no description, so the type field is
                    # actually the description.
                    desc = [paragraph([type_or_description])]
                elif raw_description:
                    assert isinstance(raw_description, list)
                    type_ = type_or_description
                    desc = [paragraph(raw_description)]
                else:
                    type_ = type_or_description
                    desc = []

                sai = SeeAlsoItem(Ref(name, None, None), desc, type_)
                new_see_also.append(sai)
                del desc
                del type_
        except Exception as e:
            raise ValueError(
                f"Error {qa}: {see_also=} | {name_and_types=}  | {raw_description=}"
            ) from e
    return new_see_also


class Gen:
    """
    Core class to generate docbundles for a given library.

    This is responsible for finding all objects, extracting the doc, parsing it,
    and saving that into the right folder.

    """

    def __init__(self, dummy_progress, config):

        if dummy_progress:
            self.Progress = DummyP
        else:
            self.Progress = Progress
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
        )

        self.log = logging.getLogger("papyri")
        self.config = config
        self.log.debug("Configuration: %s", self.config)

        self.data = {}
        self.bdata = {}
        self.metadata = {}
        self.examples = {}
        self.docs = {}

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
        for p in path.glob("**/*.rst"):
            assert p.is_file()
            parts = p.relative_to(path).parts
            assert parts[-1].endswith("rst")
            try:
                data = ts.parse(p.read_bytes())
            except Exception as e:
                raise type(e)(f"{p=}")
            blob = DocBlob()
            blob.arbitrary = data
            blob.content = {}

            blob.ordered_sections = []
            blob.item_file = None
            blob.item_line = None
            blob.item_type = None
            blob.aliases = []
            blob.example_section_data = Section()
            blob.see_also = []
            blob.signature = None
            blob.references = None
            blob.refs = []

            self.docs[parts] = json.dumps(blob.to_json(), indent=2, sort_keys=True)
            # data = p.read_bytes()

    def write_narrative(self, where: Path) -> None:
        (where / "docs").mkdir(exist_ok=True)
        for k, v in self.docs.items():
            subf = where / "docs"
            file = k[-1].rsplit(".", maxsplit=1)[0]
            file = ":".join(k[:-1]) + ":" + file
            subf.mkdir(exist_ok=True, parents=True)
            with (subf / file).open("w") as f:
                f.write(v)

    def write_examples(self, where: Path) -> None:
        (where / "examples").mkdir(exist_ok=True)
        for k, v in self.examples.items():
            with (where / "examples" / k).open("w") as f:
                f.write(v)

    def write_api(self, where: Path):
        """
        write the API section of the docbundles.
        """
        (where / "module").mkdir(exist_ok=True)
        for k, v in self.data.items():
            with (where / "module" / k).open("w") as f:
                f.write(v)

    def write(self, where: Path):
        """
        Write a docbundle folder.
        """
        self.write_api(where)
        self.write_narrative(where)
        self.write_examples(where)
        self.write_assets(where)
        with (where / "papyri.json").open("w") as f:
            f.write(json.dumps(self.metadata, indent=2, sort_keys=True))

    def write_assets(self, where: Path) -> None:
        assets = where / "assets"
        assets.mkdir()
        for k, v in self.bdata.items():
            with (assets / k).open("wb") as f:
                f.write(v)

    def put(self, path: str, data):
        """
        put some json data at the given path
        """
        self.data[path + ".json"] = data

    def put_raw(self, path: str, data):
        """
        put some rbinary data at the given path.
        """
        self.bdata[path] = data

    def prepare_doc_for_one_object(
        self,
        target_item: Any,
        ndoc,
        *,
        qa: str,
        config: Config,
        aliases: List[str],
        api_object,
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

        Returns
        -------
        Tuple of two items,
        ndoc:
            DocBundle with info for current object.
        figs:
            dict mapping figure names to figure data.

        See Also
        --------
        collect_api_docs
        """
        assert isinstance(aliases, list)
        blob = DocBlob()

        blob.content = {k: v for k, v in ndoc._parsed_data.items()}

        item_file = None
        item_line = None
        item_type = None

        # that is not going to be the case because we fallback on execution failure.

        # try to find relative path WRT site package.
        # will not work for dev install. Maybe an option to set the root location ?
        item_file = find_file(target_item)
        if item_file is not None:
            for s in SITE_PACKAGE + [os.path.expanduser("~")]:
                if item_file.startswith(s):
                    item_file = item_file[len(s) :]
        else:
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

                self.log.warn(
                    "Could not find source file for %s (%s) [%s], will not be able to link to it.",
                    repr(qa),
                    target_item,
                    type(target_item).__name__,
                )

        item_type = str(type(target_item))
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
        sig: Optional[str]
        if not blob.content["Signature"]:
            try:
                sig = str(inspect.signature(target_item))
                sig = qa.split(".")[-1] + sig
                sig = re.sub("at 0x[0-9a-f]+", "at 0x0000000", sig)
            except (ValueError, TypeError):
                sig = None
            blob.content["Signature"] = sig

        if api_object.special("Examples"):
            # warnings this is true only for non-modules
            # things.
            try:
                example_section_data, figs = get_example_data(
                    api_object.special("Examples").value,
                    obj=target_item,
                    qa=qa,
                    config=config,
                    log=self.log,
                )
            except Exception as e:
                example_section_data = Section()
                self.log.error("Error getting example data in %s", repr(qa))
                raise ValueError(f"Error getting example data in {qa!r}") from e
        else:
            example_section_data = Section()
            figs = []

        refs_2 = list(
            {
                u[1]
                for span in example_section_data
                if span.__class__.__name__ == "Code"
                for u in span.entries
                if u[1]
            }
        )
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
                refs_Ib.append(sa.name.name)

        if api_object.kind != "module":
            # TODO: most module docstring are not properly parsed by numpydoc.
            # but some are.
            assert refs_I == refs_Ib, (refs_I, refs_Ib)

        blob.example_section_data = example_section_data
        blob.refs = [normalise_ref(r) for r in sorted(set(refs_I + refs_2))]

        blob.ordered_sections = ndoc.ordered_sections
        blob.item_file = item_file
        blob.item_line = item_line
        blob.item_type = item_type

        blob.signature = blob.content.pop("Signature")
        blob.references = blob.content.pop("References")

        del blob.content["Examples"]
        del blob.content["index"]

        if blob.references == "":
            # TODO:fix
            blob.references = None
        blob.aliases = aliases
        for section in ["Extended Summary", "Summary", "Notes", "Warnings"]:
            try:
                data = blob.content.get(section, None)
                if data is None:
                    # don't exists
                    pass
                elif not data:
                    # is empty
                    blob.content[section] = Section()
                else:
                    tsc = ts.parse("\n".join(data).encode())
                    assert len(tsc) in (0, 1), (tsc, data)
                    if tsc:
                        tssc = tsc[0]
                    else:
                        tssc = Section()
                    assert isinstance(tssc, Section)
                    blob.content[section] = tssc
            except Exception:
                self.log.exception(f"Skipping section {section!r} in {qa!r} (Error)")
                raise
        assert isinstance(blob.content["Summary"], Section)
        assert isinstance(
            blob.content.get("Summary", Section()), Section
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
            new_content = Section()
            for param, type_, desc in blob.content[s]:
                assert isinstance(desc, list)
                items = []
                if desc:
                    try:
                        items = parse_rst_section("\n".join(desc))
                    except Exception as e:
                        raise type(e)(f"from {qa}")
                    for l in items:
                        assert not isinstance(l, Section)
                new_content.append(Param(param, type_, desc=items).validate())
            blob.content[s] = new_content

        blob.see_also = _normalize_see_also(blob.content.get("See Also", None), qa)
        del blob.content["See Also"]
        return blob, figs

    def collect_examples(self, folder, config):
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

        p = lambda: self.Progress(
            TextColumn("[progress.description]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "[progress.completed]{task.completed} / {task.total}",
            TimeElapsedColumn(),
        )
        with p() as p2:
            failed = []
            taskp = p2.add_task(description="Collecting examples", total=len(examples))
            for example in examples:
                p2.update(taskp, description=str(example).ljust(7))
                p2.advance(taskp)
                executor = BlockExecutor({})
                script = example.read_text()
                ce_status = "None"
                figs = []
                if config.exec:
                    with executor:
                        try:
                            executor.exec(script)
                            print(script)
                            figs = [
                                (f"ex-{example.name}-{i}.png", f)
                                for i, f in enumerate(executor.get_figs())
                            ]
                            ce_status = "execed"
                        except Exception as e:
                            failed.append(str(example))
                            if config.exec_failure == "fallback":
                                self.log.error("%s failed %s", example, type(e))
                            else:
                                raise type(e)(f"Within {example}")
                entries = parse_script(
                    script,
                    ns={},
                    prev="",
                    config=config,
                )
                s = Section(
                    [Code(entries, "", ce_status)] + [Fig(name) for name, _ in figs]
                )
                s = processed_example_data(s)

                acc.append(
                    (
                        {example.name: s},
                        figs,
                    )
                )
        assert len(failed) == 0, failed
        return acc

    def _get_collector(self):
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
                self.examples.update(
                    {
                        k: json.dumps(v.to_json(), indent=2, sort_keys=True)
                        for k, v in edoc.items()
                    }
                )
                for name, data in figs:
                    self.put_raw(name, data)

    def helper_1(self, *, qa: str, target_item, failure_collection):
        """
        Parameters
        ----------
        qa : str
            fully qualified name of the object we are extracting the
            documentation from .
        """
        item_docstring = target_item.__doc__
        builtin_function_or_method = type(sum)

        if isinstance(target_item, ModuleType):
            api_object = APIObjectInfo("module", target_item.__doc__)
        elif isinstance(target_item, (FunctionType, builtin_function_or_method)):
            api_object = APIObjectInfo("function", target_item.__doc__)
        elif isinstance(target_item, type):
            api_object = APIObjectInfo("class", target_item.__doc__)
        else:
            api_object = APIObjectInfo("other", target_item.__doc__)
            # print("Other", target_item)
            # assert False, type(target_item)

        if item_docstring is None and not isinstance(target_item, ModuleType):
            return None, None, None

        elif item_docstring is None and isinstance(target_item, ModuleType):
            item_docstring = """This module has no documentation"""
        try:
            sections = ts.parse(dedent_but_first(item_docstring).encode())
        except (AssertionError, NotImplementedError) as e:
            self.log.error("TS could not parse %s, %s", repr(qa), e)
            raise type(e)(f"from {qa}") from e
            sections = []
        except Exception as e:
            raise type(e)(f"from {qa}")

        return item_docstring, sections, api_object

    def collect_package_metadata(self, root, relative_dir):
        """
        Try to gather generic metadata about the current package we are going to
        build the documentation for.
        """
        self.root = root
        if self.config.logo:
            self.put_raw(
                "logo.png", (relative_dir / Path(self.config.logo)).read_bytes()
            )

        module = __import__(root)
        self.version = module.__version__

    def collect_api_docs(
        self,
        root: str,
    ):
        """
        Crawl one module and stores resulting docbundle in self.store.

        Parameters
        ----------
        root : str
            module name to generate docbundle for.

        See Also
        --------
        prepare_doc_for_one_object

        """

        p = lambda: self.Progress(
            TextColumn("[progress.description]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "[progress.completed]{task.completed} / {task.total}",
            TimeElapsedColumn(),
        )

        collector = self._get_collector()
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
                json.dumps(missing, indent=2, sort_keys=True),
            )

        collected = {k: v for k, v in collected.items() if k not in excluded}

        known_refs = frozenset(
            {RefInfo(root, self.version, "module", qa) for qa in collected.keys()}
        )

        with p() as p2:

            # just nice display of progression.
            taskp = p2.add_task(description="parsing", total=len(collected))

            failure_collection: Dict[str, List[str]] = defaultdict(lambda: [])

            for qa, target_item in collected.items():
                p2.update(taskp, description=qa)
                p2.advance(taskp)

                try:
                    with with_context(qa=qa):
                        item_docstring, arbitrary, api_object = self.helper_1(
                            qa=qa,
                            target_item=target_item,
                            # mutable, not great.
                            failure_collection=failure_collection,
                        )
                except Exception as e:
                    failure_collection["ErrorHelper1-" + str(type(e))].append(qa)
                    # raise
                    # continue

                try:
                    if item_docstring is None:
                        continue
                    else:
                        ndoc = NumpyDocString(dedent_but_first(item_docstring))
                        # note currentlu in ndoc we use:
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
                ex = self.config.exec
                if self.config.exec and any(
                    qa.startswith(pat) for pat in self.config.execute_exclude_patterns
                ):
                    ex = False
                dv = DirectiveVisiter(qa, known_refs, local_refs={}, aliases={})

                try:
                    # TODO: ndoc-placeholder : make sure ndoc placeholder handled here.
                    doc_blob, figs = self.prepare_doc_for_one_object(
                        target_item,
                        ndoc,
                        qa=qa,
                        config=self.config.replace(exec=ex),
                        aliases=collector.aliases[qa],
                        api_object=api_object,
                    )
                    doc_blob.arbitrary = [dv.visit(s) for s in arbitrary]
                except Exception as e:
                    self.log.error("Execution error in %s", repr(qa))
                    failure_collection["ExecError-" + str(type(e))].append(qa)
                    # continue
                    raise

                doc_blob.example_section_data = dv.visit(doc_blob.example_section_data)

                # eg, dask: str, dask.array.gufunc.apply_gufun: List[str]
                assert isinstance(doc_blob.references, (list, str, type(None))), (
                    repr(doc_blob.references),
                    qa,
                )

                if isinstance(doc_blob.references, str):
                    print(repr(doc_blob.references))
                doc_blob.references = None

                # end processing
                try:
                    doc_blob.validate()
                except Exception as e:
                    raise type(e)(f"Error in {qa}")
                self.put(qa, json.dumps(doc_blob.to_json(), indent=2, sort_keys=True))
                for name, data in figs:
                    self.put_raw(name, data)
            if failure_collection:
                self.log.info(
                    "The following parsing failed \n%s",
                    json.dumps(failure_collection, indent=2, sort_keys=True),
                )
            found = {}
            not_found = []
            for k, v in collector.aliases.items():
                if [item for item in v if item != k]:
                    if shorter := find_cannonical(k, v):
                        found[k] = shorter
                    else:
                        not_found.append((k, v))

            self.metadata = {
                "version": self.version,
                "logo": "logo.png",
                "aliases": found,
                "module": self.root,
            }


def is_private(path):
    """
    Determine if a import path, or fully qualified is private.
    that usually implies that (one of) the path part starts with a single underscore.
    """
    for p in path.split("."):
        if p.startswith("_") and not p.startswith("__"):
            return True
    return False


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
    qa_level = qa.count(".")
    min_alias_level = min(a.count(".") for a in set(aliases))
    if min_alias_level < qa_level:
        shorter_candidates = [c for c in aliases if c.count(".") <= min_alias_level]
    else:
        shorter_candidates = [c for c in aliases if c.count(".") <= qa_level]

    if (
        len(shorter_candidates) == 1
        and not is_private(shorter_candidates[0])
        and shorter_candidates[0] != qa
    ):
        return shorter_candidates[0]
    return None
