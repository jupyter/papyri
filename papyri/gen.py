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
import site
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import jedi
import toml
from IPython.core.oinspect import find_file
from pygments import lex
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, ProgressColumn
from rich.progress import Text as RichText
from rich.progress import TextColumn
from there import print
from velin.examples_section_utils import InOut, splitblank, splitcode

from .miscs import BlockExecutor, DummyP
from .take2 import (
    Code,
    Fig,
    Lines,
    Node,
    Param,
    Ref,
    RefInfo,
    Section,
    SeeAlsoItem,
    Text,
    make_block_3,
    parse_rst_to_papyri_tree,
)
from .tree import DirectiveVisiter
from .utils import dedent_but_first, pos_to_nl, progress
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


def paragraph(lines) -> Any:
    """
    Leftover rst parsing,

    Remove at some point.
    """
    [section] = ts.parse("\n".join(lines).encode())
    assert len(section.children) == 1
    p2 = section.children[0]
    return p2


def paragraphs(lines) -> List[Any]:
    assert isinstance(lines, list)
    for l in lines:
        if isinstance(l, str):
            assert "\n" not in l
        else:
            assert "\n" not in l._line
    blocks_data = make_block_3(Lines(lines))
    acc = []

    # blocks_data = parse_rst_to_papyri_tree("\n".join(lines))

    # for pre_blank_lines, blank_lines, post_black_lines in blocks_data:
    for pre_blank_lines, _blank_lines, post_blank_lines in blocks_data:
        # pre_blank_lines = block.lines
        # blank_lines = block.wh
        # post_black_lines = block.ind
        if pre_blank_lines:
            acc.append(paragraph([x._line for x in pre_blank_lines]))
        ## definitively wrong but will do for now, should likely be verbatim, or recurse ?
        if post_blank_lines:
            acc.append(paragraph([x._line for x in post_blank_lines]))
        # print(block)
    return acc


def parse_script(script, ns, prev, new_config):
    """
    Parse a script into tokens and use Jedi to infer the fully qualified names
    of each token.

    Parameters
    ----------
    script : str
        the script to tokenize and infer types on
    ns : dict
        extra namespace to use with jedi's Interpreter.
    prev : str
        previous lines that lead to this.

    Yields
    ------
    index
        index in the tokenstream
    type
        pygments token type
    text
        text of the token
    reference : str
        fully qualified name of the type of current token

    """
    jeds = []

    warnings.simplefilter("ignore", UserWarning)

    l_delta = len(prev.split("\n"))
    contextscript = prev + "\n" + script
    if ns:
        jeds.append(jedi.Interpreter(contextscript, namespaces=[ns]))
    jeds.append(jedi.Script(prev + "\n" + script))
    P = PythonLexer()

    for index, _type, text in P.get_tokens_unprocessed(script):
        line_n, col_n = pos_to_nl(script, index)
        line_n += l_delta
        try:
            ref = None
            for jed in jeds:
                failed = ""
                try:
                    if (
                        new_config.infer
                        and (text not in (" .=()[],"))
                        and text.isidentifier()
                    ):
                        inf = jed.infer(line_n + 1, col_n)
                        if inf:
                            # TODO: we might want the qualname to be module_name:name for disambiguation.
                            ref = inf[0].full_name
                            # if ref.startswith('builtins'):
                            #    ref = ''
                    else:
                        ref = ""
                except (AttributeError, TypeError) as e:
                    raise type(e)(
                        f"{contextscript}, {line_n=}, {col_n=}, {prev=}, {jed=}"
                    ) from e
                    failed = "(jedi failed inference)"
                    print("failed inference on ", script, ns, jed, col_n, line_n + 1)
                break
        except IndexError:
            raise
            ref = ""
        yield text + failed, ref
    warnings.simplefilter("default", UserWarning)


def get_example_data(doc, *, obj, qa: str, new_config):
    """Extract example section data from a NumpyDocstring

    One of the section in numpydoc is "examples" that usually consist of number
    if paragraph, interleaved with examples starting with >>> and ...,

    This attempt to parse this into structured data, with text, input and output
    as well as to infer the types of each token in the input examples.

    This is currently relatively limited as the inference does not work across
    code blocks.

    Parameters
    ----------
    doc
        a docstring parsed into a NnumpyDoc document.

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

    """
    assert qa is not None
    blocks = list(map(splitcode, splitblank(doc["Examples"])))
    example_section_data = Section()
    import matplotlib.pyplot as plt
    import numpy as np

    acc = ""

    counter = 0
    ns = {"np": np, "plt": plt, obj.__name__: obj}
    executor = BlockExecutor(ns)
    figs = []
    # fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
    fig_managers = executor.fig_man()
    assert (len(fig_managers)) == 0, f"init fail in {qa} {len(fig_managers)}"
    wait_for_show = new_config.wait_for_plt_show
    with executor:
        for b in blocks:
            for item in b:
                if isinstance(item, InOut):
                    script = "\n".join(item.in_)
                    figname = None
                    ce_status = "None"
                    try:
                        compile(script, "<>", "exec")
                        ce_status = "compiled"
                    except SyntaxError:
                        ce_status = "syntax_error"
                        pass
                    raise_in_fig = None
                    did_except = False
                    if new_config.exec and ce_status == "compiled":
                        try:
                            if not wait_for_show:
                                assert len(fig_managers) == 0
                            try:
                                res, fig_managers = executor.exec(script)
                                ce_status = "execed"
                            except Exception:
                                ce_status = "exception_in_exec"
                                if new_config.exec_failure != "fallback":
                                    raise
                            if fig_managers and (
                                ("plt.show" in script) or not wait_for_show
                            ):
                                raise_in_fig = True
                                for fig in executor.get_figs():
                                    counter += 1
                                    figname = f"fig-{qa}-{counter}.png"
                                    figs.append((figname, fig))
                                plt.close("all")
                                raise_in_fig = False

                        except Exception:
                            did_except = True
                            print(f"exception executing... {qa}")
                            fig_managers = executor.fig_man()
                            if raise_in_fig:
                                raise
                        finally:
                            if not wait_for_show:
                                if fig_managers:
                                    for fig in executor.get_figs():
                                        counter += 1
                                        figname = f"fig-{qa}-{counter}.png"
                                        figs.append((figname, fig))
                                        print(
                                            f"Still fig manager(s) open for {qa}: {figname}"
                                        )
                                    plt.close("all")
                                fig_managers = executor.fig_man()
                                assert len(fig_managers) == 0, fig_managers + [
                                    did_except,
                                ]
                    infer_exclude = new_config.exclude_jedi
                    if qa in infer_exclude:
                        print(f"Turning off type inference for func {qa!r}")
                        inf = False
                    else:
                        inf = new_config.infer
                    entries = list(
                        parse_script(
                            script,
                            ns=ns,
                            prev=acc,
                            new_config=new_config.replace(infer=inf),
                        )
                    )
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
    list(lex(code, PythonLexer()))
    FMT = HtmlFormatter()
    classes = [FMT.ttype2class.get(x) for x, y in lex(code, PythonLexer())]
    classes = [c if c is not None else "" for c in classes]
    return classes


def P2(lines) -> List[Node]:
    assert isinstance(lines, list)
    for l in lines:
        if isinstance(l, str):
            assert "\n" not in l
        else:
            assert "\n" not in l._line
    assert lines, lines
    blocks_data = parse_rst_to_papyri_tree("\n".join(lines))

    # for pre_blank_lines, blank_lines, post_black_lines in blocks_data:
    for block in blocks_data:
        assert not block.__class__.__name__ == "Block", block
    return blocks_data


def processed_example_data(example_section_data):
    """this should be no-op on already ingested"""
    new_example_section_data = Section()
    for in_out in example_section_data:
        type_ = in_out.__class__.__name__
        # color examples with pygments classes
        if type_ == "Text":
            blocks = P2(in_out.value.split("\n"))
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
    dummy_progress: bool = False
    exec_failure: Optional[str] = None  # should move to enum
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


def gen_main(infer, exec_, target_file, experimental, debug, *, dummy_progress: bool):
    """
    main entry point
    """
    conffile = Path(target_file).expanduser()
    if conffile.exists():
        conf: MutableMapping[str, Any] = toml.loads(conffile.read_text())
        k0 = next(iter(conf.keys()))
        new_config = Config(**conf[k0], dummy_progress=dummy_progress)
        if exec_ is not None:
            new_config.exec = exec_
        if infer is not None:
            new_config.infer = infer

        if len(conf.keys()) != 1:
            raise ValueError(
                f"We only support one library at a time for now {conf.keys()}"
            )

        names = list(conf.keys())

    else:
        sys.exit(f"{conffile!r} does not exists.")

    tp = os.path.expanduser("~/.papyri/data")

    target_dir = Path(tp).expanduser()
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    g = Gen(
        dummy_progress=dummy_progress,
    )
    g.log.info("Will write data to %s", target_dir)
    if debug:
        g.log.setLevel("DEBUG")
        g.log.debug("Log level set to debug")

    g.do_one_mod(
        names,
        relative_dir=Path(target_file).parent,
        experimental=experimental,
        new_config=new_config,
    )
    docs_path: Optional[str] = new_config.docs_path
    if docs_path is not None:
        path = Path(docs_path).expanduser()
        g.do_docs(path)
    p = target_dir / (g.root + "_" + g.version)
    p.mkdir(exist_ok=True)

    g.log.info("Saving current Doc bundle to %s", p)
    g.clean(p)
    g.write(p)


class TimeElapsedColumn(ProgressColumn):
    """Renders estimated time remaining."""

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def render(self, task) -> RichText:
        """Show time remaining."""

        ctime = task.fields.get("ctime", None)
        if ctime is None:
            return RichText("-:--:--", style="progress.remaining")
        ctime_delta = timedelta(seconds=int(ctime))
        return RichText(
            str(ctime_delta), style="progress.remaining", overflow="ellipsis"
        )


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
        root:
            Base object, typically module we want to scan itself.
            We will attempt to no scan any object which does not belong
            to the root or one of its children.

        others:
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

        Note
        ----

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


class Gen:
    """
    Core class to generate docbundles for a given library.

    This is responsible for finding all objects, extracting the doc, parsing it,
    and saving that into the right folder.

    """

    def __init__(self, dummy_progress):

        if dummy_progress:
            self.Progress = DummyP
        else:
            self.Progress = Progress
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
        )

        self.log = logging.getLogger("papyri")

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

    def do_docs(self, path):
        """
        Crawl the filesystem for all docs/rst files

        """
        self.log.info("Scraping Documentation")
        for p in path.glob("**/*.rst"):
            assert p.is_file()
            parts = p.relative_to(path).parts
            assert parts[-1].endswith("rst")

            data = ts.parse(p.read_bytes())
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

    def write(self, where: Path):
        """
        Write a docbundle folder.
        """
        (where / "module").mkdir(exist_ok=True)
        for k, v in self.data.items():
            with (where / "module" / k).open("w") as f:
                f.write(v)

        (where / "docs").mkdir(exist_ok=True)
        for k, v in self.docs.items():
            subf = where / "docs"
            for s in k[:-1]:
                subf = subf / s
            file = k[-1]
            subf.mkdir(exist_ok=True)
            with (subf / file).open("w") as f:
                f.write(v)

        (where / "examples").mkdir(exist_ok=True)
        for k, v in self.examples.items():
            with (where / "examples" / k).open("w") as f:
                f.write(v)

        assets = where / "assets"
        assets.mkdir()
        for k, v in self.bdata.items():
            with (assets / k).open("wb") as f:
                f.write(v)

        with (where / "papyri.json").open("w") as f:
            f.write(json.dumps(self.metadata, indent=2, sort_keys=True))

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

    def do_one_item(
        self,
        target_item: Any,
        ndoc,
        *,
        qa: str,
        new_config,
    ) -> Tuple[DocBlob, List]:
        """
        Get documentation information for one item

        Returns
        -------
        Tuple of two items,
        ndoc:
            DocBundle with info for current object.
        figs:
            dict mapping figure names to figure data.

        See Also
        --------
        do_one_mod
        """
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

        if not blob.content["Signature"]:
            sig = None
            try:
                sig = str(inspect.signature(target_item))
                sig = qa.split(".")[-1] + sig
            except (ValueError, TypeError):
                pass
            # mutate argument ! BAD
            blob.content["Signature"] = sig

        new_see_also = ndoc["See Also"]
        refs = []
        if new_see_also:
            for line in new_see_also:
                rt, desc = line
                assert isinstance(desc, list), line
                for ref, _type in rt:
                    refs.append(ref)

        try:
            ndoc.example_section_data, figs = get_example_data(
                ndoc,
                obj=target_item,
                qa=qa,
                new_config=new_config,
            )
            ndoc.figs = figs
        except Exception as e:
            ndoc.example_section_data = Section()
            self.log.error("Error getting example data in %s", repr(qa))
            raise ValueError("Error getting example data in {qa!r}") from e
            ndoc.figs = []

        ndoc.refs = list(
            {
                u[1]
                for span in ndoc.example_section_data
                if span.__class__.__name__ == "Code"
                for u in span.entries
                if u[1]
            }
        )

        blob.example_section_data = ndoc.example_section_data
        ndoc.refs.extend(refs)
        ndoc.refs = [normalise_ref(r) for r in sorted(set(ndoc.refs))]
        figs = ndoc.figs
        del ndoc.figs

        blob.ordered_sections = ndoc.ordered_sections
        blob.refs = ndoc.refs
        blob.item_file = item_file
        blob.item_line = item_line
        blob.item_type = item_type

        return blob, figs

    def collect_examples(self, folder, new_config):
        acc = []
        examples = list(folder.glob("**/*.py"))

        valid_examples = []
        for e in examples:
            if any(str(e).endswith(p) for p in new_config.examples_exclude):
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
                if new_config.exec:
                    with executor:
                        try:
                            executor.exec(script)
                            figs = [
                                (f"ex-{example.name}-{i}.png", f)
                                for i, f in enumerate(executor.get_figs())
                            ]
                            ce_status = "execed"
                        except Exception as e:
                            self.log.error("%s failed %s", example, type(e))
                            failed.append(str(example))
                            continue
                            # raise type(e)(f"Within {example}")
                entries = list(
                    parse_script(
                        script,
                        ns={},
                        prev="",
                        new_config=new_config,
                    )
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

    def configure(self, names: List[str], new_config):
        """
        Configure current instance of gen

        Parameters
        ----------

        names: List of str
            modules and submodules to recursively crawl.
            The first one is assumed to be the root, others, submodules not
            reachable from the root.

        """
        assert len(names) == 1
        modules = []
        for name in names:
            x, *r = name.split(".")
            n0 = __import__(name)
            for sub in r:
                n0 = getattr(n0, sub)
            modules.append(n0)

        root = names[0].split(".")[0]
        self.root = root

        # step 2 try to guess the version number from the top module.
        version = getattr(modules[0], "__version__", "???")
        self.version = version

        subs = new_config.submodules
        extra_from_conf = [self.root + "." + s for s in subs]
        for name in extra_from_conf:
            x, *r = name.split(".")
            n0 = __import__(name)
            for sub in r:
                n0 = getattr(n0, sub)
            modules.append(n0)

        collector = DFSCollector(modules[0], modules[1:])

        return collector

    def collect_examples_out(self, new_config):

        examples_folder = new_config.examples_folder
        self.log.debug("Example Folder: %s", examples_folder)
        if examples_folder is not None:
            examples_folder = Path(examples_folder).expanduser()
            examples_data = self.collect_examples(
                examples_folder,
                new_config=new_config,
            )
            for edoc, figs in examples_data:
                self.examples.update(
                    {
                        k: json.dumps(v.to_json(), indent=2, sort_keys=True)
                        for k, v in edoc.items()
                    }
                )
                for name, data in figs:
                    print("put one fig", name)
                    self.put_raw(name, data)

    def helper_1(self, *, qa: str, experimental: bool, target_item, failure_collection):
        """
        Parameters
        ----------
        qa : str
            fully qualified name of the object we are extracting the
        documentation from .
        experimental : bool
            whether to try experimental features
        p2: rich progress instance
        """
        short_description = (qa[:19] + "..") if len(qa) > 21 else qa
        item_docstring = target_item.__doc__
        # TODO: we may not want to skip items as they may have children
        # right now keep modules, but we may want to keep classes if
        # they have documented descendants.

        if item_docstring is None and not isinstance(target_item, ModuleType):
            return short_description, None, None

        elif item_docstring is None and isinstance(target_item, ModuleType):
            item_docstring = """This module has no documentation"""
        try:
            arbitrary = ts.parse(dedent_but_first(item_docstring).encode())
        except (AssertionError, NotImplementedError) as e:
            self.log.warning("TS could not parse %s, %s", repr(qa), e)
            failure_collection[type(e).__name__].append(qa)
            if experimental:
                raise type(e)(f"from {qa}") from e
            arbitrary = []
        except Exception as e:
            raise type(e)(f"from {qa}")

        return short_description, item_docstring, arbitrary

    def do_one_mod(
        self,
        names: List[str],
        relative_dir: Path,
        *,
        experimental,
        new_config: Config,
    ):
        """
        Crawl one modules and stores resulting docbundle in self.store.

        Parameters
        ----------
        names : List[str]
            list of (sub)modules names to generate docbundle for.
            The first is considered the root module.
        exec_ : bool
            Whether to try to execute the code blocks and embed resulting values like plots.

        See Also
        --------
        do_one_item, do_one_iteration

        """

        p = lambda: self.Progress(
            TextColumn("[progress.description]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "[progress.completed]{task.completed} / {task.total}",
            TimeElapsedColumn(),
        )

        collector = self.configure(names, new_config)
        collected: Dict[str, Any] = collector.items()

        self.log.debug("Configuration: %s", new_config)

        self.collect_examples_out(new_config)

        if new_config.logo:
            self.put_raw(
                "logo.png", (relative_dir / Path(new_config.logo)).read_bytes()
            )

        # collect all items we want to document.
        excluded = sorted(new_config.exclude)
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
            {RefInfo(names[0], self.version, "module", qa) for qa in collected.keys()}
        )

        with p() as p2:

            # just nice display of progression.
            taskp = p2.add_task(description="parsing", total=len(collected))

            failure_collection: Dict[str, List[str]] = defaultdict(lambda: [])

            for qa, target_item in collected.items():
                dv = DirectiveVisiter(qa, known_refs, local_refs={}, aliases={})
                short_description, item_docstring, arbitrary = self.helper_1(
                    qa=qa,
                    experimental=experimental,
                    target_item=target_item,
                    # mutable, not great.
                    failure_collection=failure_collection,
                )
                p2.update(taskp, description=short_description.ljust(17))
                p2.advance(taskp)

                if item_docstring is None:
                    continue

                try:
                    ndoc = NumpyDocString(dedent_but_first(item_docstring))
                except Exception:
                    if not isinstance(target_item, ModuleType):
                        self.log.exception(
                            "Unexpected error parsing %s – %s",
                            qa,
                            target_item.__name__,
                        )
                    if isinstance(target_item, ModuleType):
                        # TODO: ndoc-placeholder : remove placeholder here
                        ndoc = NumpyDocString(f"To remove in the future –– {qa}")
                    else:
                        continue
                if not isinstance(target_item, ModuleType):
                    arbitrary = []
                ex = new_config.exec
                if new_config.exec and any(
                    qa.startswith(pat) for pat in new_config.execute_exclude_patterns
                ):
                    ex = False

                try:
                    # TODO: ndoc-placeholder : make sure ndoc placeholder handled here.
                    doc_blob, figs = self.do_one_item(
                        target_item,
                        ndoc,
                        qa=qa,
                        new_config=new_config.replace(exec=ex),
                    )
                    doc_blob.arbitrary = [dv.visit(s) for s in arbitrary]
                except Exception as e:
                    self.log.error("Execution error in %s", repr(qa))
                    failure_collection["ExecError-" + str(type(e))].append(qa)
                    if new_config.exec_failure == "fallback":
                        print("Re-analysing ", qa, "without execution", type(e))
                        # debug:
                        # TODO: ndoc-placeholder : make sure ndoc placeholder handled here as well.
                        try:
                            doc_blob, figs = self.do_one_item(
                                target_item,
                                ndoc,
                                qa=qa,
                                new_config=new_config.replace(exec=False),
                            )
                            doc_blob.arbitrary = [dv.visit(s) for s in arbitrary]
                        except Exception as e:
                            self.log.exception(
                                "unexpected non-exec error in %s", repr(qa)
                            )
                            failure_collection["ErrorNoExec-" + str(type(e))].append(qa)
                            continue
                doc_blob.example_section_data = dv.visit(doc_blob.example_section_data)
                doc_blob.aliases = collector.aliases[qa]

                # processing....
                doc_blob.signature = doc_blob.content.pop("Signature")

                ## TODO: here type instability of Summary, and other stuff convert before.
                for section in ["Extended Summary", "Summary", "Notes", "Warnings"]:
                    try:
                        if section in doc_blob.content:
                            if data := doc_blob.content[section]:
                                # assert (
                                #    False
                                # ), "will get 'Was not able to parse docstring for ...'"
                                # TODO : the following is in progress as we try to move away from custom parsing and use
                                # tree_ssitter.
                                tsc = ts.parse("\n".join(data).encode())
                                assert len(tsc) in (0, 1), (tsc, data)
                                if tsc:
                                    tssc = tsc[0]
                                else:
                                    tssc = Section()
                                assert isinstance(tssc, Section)
                                doc_blob.content[section] = tssc
                            else:
                                doc_blob.content[section] = Section()
                    except Exception as e:
                        self.log.exception(
                            f"Skipping section {section!r} in {qa!r} (Error)"
                        )
                        failure_collection[type(e).__name__].append(qa)
                        doc_blob.content[section] = ts.parse(
                            b"Parsing not NotImplemented for this section."
                        )[0]
                        if experimental:
                            raise type(e)(f"during {qa}") from e

                if not isinstance(doc_blob.content["Summary"], Section):
                    assert isinstance(doc_blob.content["Summary"], list)
                    assert len(doc_blob.content["Summary"]) == 1
                    # doc_blob.content["Summary"] = ts.parse(

                    for s in doc_blob.content["Summary"]:
                        assert isinstance(s, str)

                if "Summary" in doc_blob.content:
                    assert isinstance(
                        doc_blob.content["Summary"], Section
                    ), doc_blob.content["Summary"]

                doc_blob.references = doc_blob.content.pop("References")

                # eg, dask: str, dask.array.gufunc.apply_gufun: List[str]
                assert isinstance(doc_blob.references, (list, str)), (
                    repr(doc_blob.references),
                    qa,
                )

                doc_blob.references = None

                del doc_blob.content["Examples"]
                del doc_blob.content["index"]
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

                #        new_doc_blob._content["Parameters"] = [
                #            Parameter(a, b, c)
                #            for (a, b, c) in new_doc_blob._content.get("Parameters", [])
                #        ]

                for s in sections_:
                    if s in doc_blob.content:
                        assert isinstance(
                            doc_blob.content[s], list
                        ), f"{s}, {doc_blob.content[s]} "
                        new_content = Section()
                        for param, type_, desc in doc_blob.content[s]:
                            assert isinstance(desc, list)
                            items = []
                            if desc:
                                try:
                                    items = P2(desc)
                                except Exception as e:
                                    raise type(e)(f"from {qa}")
                                for l in items:
                                    assert not isinstance(l, Section)
                            new_content.append(
                                Param(param, type_, desc=items).validate()
                            )
                        doc_blob.content[s] = new_content

                doc_blob.see_also = []
                if see_also := doc_blob.content.get("See Also", None):
                    for nts, d0 in see_also:
                        try:
                            d = d0
                            for (name, type_or_description) in nts:
                                if type_or_description and not d:
                                    desc = type_or_description
                                    if isinstance(desc, str):
                                        desc = [desc]
                                    assert isinstance(desc, list)
                                    desc = paragraphs(desc)
                                    type_ = None
                                else:
                                    desc = d0
                                    type_ = type_or_description
                                    assert isinstance(desc, list)
                                    desc = paragraphs(desc)

                                sai = SeeAlsoItem(Ref(name, None, None), desc, type_)
                                doc_blob.see_also.append(sai)
                                del desc
                                del type_
                        except Exception as e:
                            raise ValueError(
                                f"Error {qa}: {see_also=}    |    {nts=}    | {d0=}"
                            ) from e
                del doc_blob.content["See Also"]

                for k, v in doc_blob.content.items():
                    assert isinstance(v, Section), f"{k} is not a section {v}"
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
