from __future__ import annotations

import inspect
import io
import json
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path

# from numpydoc.docscrape import NumpyDocString
from types import FunctionType, ModuleType
from typing import Any, Dict, List, Optional, Tuple

import jedi
from pygments import lex
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from rich.progress import BarColumn, Progress, ProgressColumn
from rich.progress import Text as RichText
from rich.progress import TextColumn
from there import print
from velin.examples_section_utils import InOut, splitblank, splitcode

from .take2 import main as t2main
from .take2 import (
    Code,
    Fig,
    Lines,
    Math,
    Node,
    Paragraph,
    Ref,
    Section,
    SeeAlsoItem,
    Text,
    make_block_3,
)
from .utils import dedent_but_first, pos_to_nl, progress
from .vref import NumpyDocString

try:
    from .ts import tsparse
except ImportError:
    print("TREE SITTER IMPORTING FAILED, will return empty on parse")
    tsparse = lambda x: []


def paragraph(lines) -> List[Tuple[str, Any]]:
    """
    return container of (type, obj)
    """
    p = Paragraph.parse_lines(lines)
    acc = []
    for c in p.children:
        if type(c).__name__ == "Directive":
            if c.role == "math":
                acc.append(Math(c.value))
            else:
                acc.append(c)
        else:
            acc.append(c)
    p.children = acc
    return p


def paragraphs(lines) -> List[Any]:
    assert isinstance(lines, list)
    for l in lines:
        if isinstance(l, str):
            assert "\n" not in l
        else:
            assert "\n" not in l._line
    blocks_data = make_block_3(Lines(lines))
    acc = []

    # blocks_data = t2main("\n".join(lines))

    # for pre_blank_lines, blank_lines, post_black_lines in blocks_data:
    for pre_blank_lines, blank_lines, post_black_lines in blocks_data:
        # pre_blank_lines = block.lines
        # blank_lines = block.wh
        # post_black_lines = block.ind
        if pre_blank_lines:
            acc.append(paragraph([x._line for x in pre_blank_lines]))
        ## definitively wrong but will do for now, should likely be verbatim, or recurse ?
        if post_black_lines:
            acc.append(paragraph([x._line for x in post_black_lines]))
        # print(block)
    return acc


def parse_script(script, ns=None, infer=None, prev="", config=None):
    """
    Parse a script into tokens and use Jedi to infer the fully qualified names
    of each token.

    Parameters
    ----------
    script : str
        the script to tokenize and infer types on
    ns : dict
        extra namespace to use with jedi's Interpreter.
    infer : bool
        whether to run jedi type inference that can be quite time consuming.
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
    import warnings

    warnings.simplefilter("ignore", UserWarning)

    l_delta = len(prev.split("\n"))
    contextscript = prev + "\n" + script
    if ns:
        jeds.append(jedi.Interpreter(contextscript, namespaces=[ns]))
    jeds.append(jedi.Script(prev + "\n" + script))
    P = PythonLexer()

    for index, type_, text in P.get_tokens_unprocessed(script):
        line_n, col_n = pos_to_nl(script, index)
        line_n += l_delta
        try:
            ref = None
            for jed in jeds:
                failed = ""
                try:
                    if infer and (text not in (" .=()[],")) and text.isidentifier():
                        inf = jed.infer(line_n + 1, col_n)
                        if inf:
                            ref = inf[0].full_name
                            # if ref.startswith('builtins'):
                            #    ref = ''
                    else:
                        ref = ""
                except (AttributeError, TypeError, Exception) as e:
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


class BlockExecutor:
    """
    To merge with next function; a block executor that
    can take sequences of code, keep state and will return the figures generated.
    """

    def __init__(self, ns):
        import matplotlib

        matplotlib.use("agg")
        self.ns = ns
        pass

    def __enter__(self):
        assert (len(self.fig_man())) == 0, f"init fail in {qa} {len(fig_managers)}"

    def __exit__(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        plt.close("all")
        assert (len(self.fig_man())) == 0, f"init fail in {qa} {len(fig_managers)}"

    def fig_man(self):
        from matplotlib import _pylab_helpers

        return _pylab_helpers.Gcf.get_all_fig_managers()

    def get_figs(self):
        figs = []
        for fig_man in self.fig_man():
            buf = io.BytesIO()
            fig_man.canvas.figure.savefig(buf, dpi=300)  # , bbox_inches="tight"
            buf.seek(0)
            figs.append(buf.read())
        return figs

    def exec(self, text):
        from matplotlib import _pylab_helpers, cbook
        from matplotlib.backend_bases import FigureManagerBase

        with cbook._setattr_cm(FigureManagerBase, show=lambda self: None):
            res = exec(text, self.ns)

        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()

        return res, fig_managers


def get_example_data(doc, infer=True, obj=None, exec_=True, qa=None, config=None):
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
    infer : bool
        whether to run type inference; which can be time consuming.

    """
    assert qa is not None
    if not config:
        config = {}
    blocks = list(map(splitcode, splitblank(doc["Examples"])))
    example_section_data = Section()
    import matplotlib.pyplot as plt
    from matplotlib import _pylab_helpers

    acc = ""
    import numpy as np

    counter = 0
    ns = {"np": np, "plt": plt, obj.__name__: obj}
    executor = BlockExecutor(ns)
    figs = []
    fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
    assert (len(fig_managers)) == 0, f"init fail in {qa} {len(fig_managers)}"
    wait_for_show = config.get("wait_for_plt_show", True)
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
                    raise_in_fig = "?"
                    did_except = False
                    if exec_:
                        try:
                            if not wait_for_show:
                                assert len(fig_managers) == 0
                            try:
                                res, fig_managers = executor.exec(script)
                                ce_status = "execed"
                            except Exception:
                                ce_status = "exception_in_exec"
                                if config.get("exec_failure", "") != "fallback":
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
                            fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
                            if raise_in_fig:
                                raise
                        finally:
                            if not wait_for_show:
                                if fig_managers:
                                    plt.close("all")
                                fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
                                assert len(fig_managers) == 0, fig_managers + [
                                    did_except,
                                ]
                    infer_exclude = config.get("exclude_jedi", frozenset())
                    if qa in infer_exclude:
                        print("Turning off type inference for this function:", qa)
                        inf = False
                    else:
                        inf = infer
                    entries = list(
                        parse_script(script, ns=ns, infer=inf, prev=acc, config=config)
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
    fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
    if len(fig_managers) != 0:
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
    blocks_data = t2main("\n".join(lines))

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


def gen_main(names, infer, exec_):
    """
    main entry point
    """
    import toml
    import os

    conffile = Path("~/.papyri/papyri.toml").expanduser()
    conf = toml.loads(conffile.read_text())

    global_conffile = Path("~/.papyri/config.toml").expanduser()
    global_conf = toml.loads(global_conffile.read_text())

    # tp = global_conf.get("global", {}).get("target_path", ".")
    tp = os.path.expanduser("~/.papyri/data")

    target_dir = Path(tp).expanduser()
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    print(target_dir)
    g = Gen()
    g.do_one_mod(names, infer, exec_, conf)
    p = target_dir / (g.root + "_" + g.version)
    p.mkdir(exist_ok=True)

    g.clean(p)
    g.write(p)


def timer(progress, task):
    c = 0

    @contextmanager
    def timeit():
        now = time.monotonic()
        yield
        nonlocal c
        c += time.monotonic() - now
        progress.update(task, ctime=c)

    return timeit


class TimeElapsedColumn(ProgressColumn):
    """Renders estimated time remaining."""

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def render(self, task) -> RichText:
        """Show time remaining."""
        from datetime import timedelta

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
    def __init__(self, root, others):
        assert isinstance(root, ModuleType), root
        self.root = root.__name__
        assert "." not in self.root
        self.obj = dict()
        self.aliases = defaultdict(lambda: [])
        self._open_list = [(root, [root.__name__])]
        for o in others:
            self._open_list.append((o, o.__name__.split(".")))

    def items(self):
        while len(self._open_list) >= 1:
            current, stack = self._open_list.pop(0)

            # numpy objects ane no bool values.
            if id(current) not in [id(x) for x in self.obj.values()]:
                self.visit(current, stack)

        return self.obj

    def visit(self, obj, stack):
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


class Collector:
    def __init__(self, root):
        assert isinstance(root, ModuleType), root
        self.root = root
        self.obj = dict()
        self.aliases = defaultdict(lambda: [])
        self.stack = [self.root.__name__]
        self._open_list = []

    def visit_ModuleType(self, mod):
        for k in dir(mod):
            self.stack.append(k)
            self.visit(getattr(mod, k))
            self.stack.pop()

    def visit_ClassType(self, klass):
        for k, v in klass.__dict__.items():
            self.stack.append(k)
            self.visit(v)
            self.stack.pop()

    def visit_FunctionType(self, fun):
        pass

    def visit(self, obj):
        pp = False
        if self.stack == ["scipy", "fft", "set_workers"]:
            pp = True
        try:
            qa = full_qual(obj)
        except Exception as e:
            raise RuntimeError(f"error visiting {'.'.join(self.stack)}") from e
        if not qa:
            return
        if not qa.startswith(self.root.__name__):
            return
        if obj in self.obj.values():
            fq = [k for k, v in self.obj.items() if obj is v][0]
            sn = ".".join(self.stack)
            if fq != sn:
                self.aliases[qa].append(sn)
            if pp:
                print("SKIP", obj, fq, qa)
            return
        if (qa in self.obj) and self.obj[qa] != obj:
            pass
        self.obj[qa] = obj

        if (sn := ".".join(self.stack)) != qa:
            self.aliases[qa].append(sn)

        if isinstance(obj, ModuleType):
            return self.visit_ModuleType(obj)
        elif isinstance(obj, FunctionType):
            return self.visit_FunctionType(obj)
        elif isinstance(obj, type):
            return self.visit_ClassType(obj)
        else:
            pass
            # print('Dont know haw to visit {type(obj)=}, {obj =}')

    def items(self):
        self.visit(self.root)
        return self.obj


from .take2 import Node


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
        arbitrary = []

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
    def __init__(self):
        self.data = {}
        self.bdata = {}
        self.metadata = {}
        self.examples = {}

    def clean(self, where: Path):
        for _, path in progress(
            (where / "module").glob("*.json"), description="cleaning previous bundle"
        ):
            path.unlink()
        for _, path in progress(
            (where / "assets").glob("*"), description="cleaning previous bundle"
        ):
            path.unlink()

        if (where / "module").exists():
            (where / "module").rmdir()
        if (where / "assets").exists():
            (where / "assets").rmdir()
        if (where / "papyri.json").exists():
            (where / "papyri.json").unlink()

    def write(self, where: Path):
        (where / "module").mkdir(exist_ok=True)
        for k, v in self.data.items():
            with (where / "module" / k).open("w") as f:
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
            f.write(json.dumps(self.metadata, indent=2))

    def put(self, path: str, data):
        self.data[path + ".json"] = data

    def put_raw(self, path: str, data):
        self.bdata[path] = data

    def do_one_item(
        self, target_item: Any, ndoc, infer: bool, exec_: bool, qa: str, config=None
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
        """
        if config is None:
            config = {}
        blob = DocBlob()

        blob.content = {k: v for k, v in ndoc._parsed_data.items()}
        item_file = None
        item_line = None
        item_type = None
        try:
            item_file = inspect.getfile(target_item)
            item_line = inspect.getsourcelines(target_item)[1]
            item_type = str(type(target_item))
        except (AttributeError, TypeError):
            pass
        except OSError:
            pass

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
                for ref, type_ in rt:
                    refs.append(ref)

        try:
            ndoc.example_section_data, figs = get_example_data(
                ndoc, infer, obj=target_item, exec_=exec_, qa=qa, config=config
            )
            ndoc.figs = figs
        except Exception as e:
            ndoc.example_section_data = Section()
            print("Error getting example data in ", qa)
            raise ValueError("Error getting example data in ", qa) from e
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

    def collect_examples(self, folder):
        acc = []
        examples = list(folder.glob("*.py"))
        for example in examples:
            executor = BlockExecutor({})
            with executor:
                script = example.read_text()
                executor.exec(script)
                figs = [
                    (f"ex-{example.name}-{i}.png", f)
                    for i, f in enumerate(executor.get_figs())
                ]
                entries = list(parse_script(script, ns={}, infer=True, prev=""))
                s = Section(
                    [Code(entries, "", "execed")] + [Fig(name) for name, _ in figs]
                )
                s = processed_example_data(s)

                acc.append(
                    (
                        {example.name: s},
                        figs,
                    )
                )
        return acc

    def do_one_mod(self, names: List[str], infer: bool, exec_: bool, conf: dict):
        """
        Crawl one modules and stores resulting docbundle in self.store.

        Parameters
        ----------
        names : List[str]
            list of (sub)modules names to generate docbundle for.
            The first is considered the root module.
        infer : bool
            Wether to run type inference with jedi.
        exec_ : bool
            Wether to try to execute the code blocks and embed resulting values like plots.
        """

        p = lambda: Progress(
            TextColumn("[progress.description]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "[progress.completed]{task.completed} / {task.total}",
            TimeElapsedColumn(),
        )
        # step one collect all the modules instances we want to analyse.

        modules = []
        for name in names:
            x, *r = name.split(".")
            n0 = __import__(name)
            for sub in r:
                n0 = getattr(n0, sub)
            modules.append(n0)

        # step 2 try to guess the version number from the top module.
        version = getattr(modules[0], "__version__", "???")

        root = names[0].split(".")[0]
        module_conf = conf.get(root, {})
        examples_folder = module_conf.get("examples_folder", None)
        print("EF", examples_folder)
        if examples_folder is not None:
            examples_folder = Path(examples_folder).expanduser()
            examples_data = self.collect_examples(examples_folder)
            for edoc, figs in examples_data:
                self.examples.update(
                    {k: json.dumps(v.to_json()) for k, v in edoc.items()}
                )
                for name, data in figs:
                    print("put one fig", name)
                    self.put_raw(name, data)
        print("Configuration:", json.dumps(module_conf, indent=2))
        self.root = root
        self.version = version
        subs = module_conf.get("submodules", [])
        extra_from_conf = [root + "." + s for s in subs]
        for name in extra_from_conf:
            x, *r = name.split(".")
            n0 = __import__(name)
            for sub in r:
                n0 = getattr(n0, sub)
            modules.append(n0)

        # print(modules)

        collector = DFSCollector(modules[0], modules[1:])
        collected: Dict[str, Any] = collector.items()

        # collect all items we want to document.
        for qa, item in collected.items():
            if (nqa := full_qual(item)) != qa:
                print("after import qa differs : {qa} -> {nqa}")
                if collected[nqa] == item:
                    print("present twice")
                    del collected[nqa]
                else:
                    print("differs: {item} != {other}")

        for target in module_conf.get("exclude", []):
            print("exclude tgt:", target)
            del collected[target]
        # p = nullcontext
        with p() as p2:

            # just nice display of progression.
            taskp = p2.add_task(description="parsing", total=len(collected))

            for qa, target_item in collected.items():
                short_description = (qa[:19] + "..") if len(qa) > 21 else qa
                p2.update(taskp, description=short_description.ljust(17))
                p2.advance(taskp)
                item_docstring = target_item.__doc__

                # TODO: we may not want tosip items as they may have children
                # right now keep modules, but we may want to keep classes if
                # they have documented descendants.

                if item_docstring is None and not isinstance(target_item, ModuleType):
                    continue
                elif item_docstring is None and isinstance(target_item, ModuleType):
                    item_docstring = """This module has no documentation"""

                # progress.console.print(qa)
                try:
                    arbitrary = tsparse(dedent_but_first(item_docstring).encode())
                except Exception as e:
                    print(f"TS could not parse: {qa}")
                    raise ValueError(f"from {qa}") from e
                    arbitrary = []
                    # raise
                try:
                    ndoc = NumpyDocString(dedent_but_first(item_docstring))
                except Exception:
                    if not isinstance(target_item, ModuleType):
                        p2.console.print(
                            "Unexpected error parsing",
                            target_item,
                            target_item.__name__,
                        )
                    if isinstance(target_item, ModuleType):
                        # from .take2 import main
                        # main(item_docstring)
                        ndoc = NumpyDocString(
                            f"Was not able to parse docstring for {qa}"
                        )
                    else:
                        continue
                if not isinstance(target_item, ModuleType):
                    arbitrary = []
                execute_exclude_patterns = module_conf.get(
                    "execute_exclude_patterns", None
                )
                ex = exec_
                if execute_exclude_patterns and exec_:
                    for pat in execute_exclude_patterns:
                        if qa.startswith(pat):
                            ex = False
                            break
                # else:
                #    print("will run", qa)

                try:
                    doc_blob, figs = self.do_one_item(
                        target_item, ndoc, infer, ex, qa, config=module_conf
                    )
                    doc_blob.arbitrary = arbitrary
                except Exception:
                    raise
                    if module_conf.get("exec_failure", None) == "fallback":
                        print("Re-analysing ", qa, "without execution")
                        # debug:
                        doc_blob, figs = self.do_one_item(
                            target_item, ndoc, infer, False, qa, config=module_conf
                        )
                doc_blob.aliases = collector.aliases[qa]

                # processing....
                doc_blob.signature = doc_blob.content.pop("Signature")
                try:
                    for section in ["Extended Summary", "Summary", "Notes", "Warnings"]:
                        if section in doc_blob.content:
                            if data := doc_blob.content[section]:
                                PX = P2(data)
                                doc_blob.content[section] = Section(PX)
                            else:
                                doc_blob.content[section] = Section()
                except Exception as e:
                    raise type(e)(f"during {qa}")

                doc_blob.references = doc_blob.content.pop("References")
                if isinstance(doc_blob.references, str):
                    if doc_blob.references == "":
                        doc_blob.references = None
                    else:
                        assert False
                        doc_blob.references = list(doc_blob.references)
                assert (
                    isinstance(doc_blob.references, list) or doc_blob.references is None
                )
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
                from .take2 import Param

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
                                items = P2(desc)
                            new_content.append(Param(param, type_, items))
                        doc_blob.content[s] = new_content

                doc_blob.see_also = []
                if (see_also := doc_blob.content.get("See Also", None)) :
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

                self.put(qa, json.dumps(doc_blob.to_json(), indent=2))
                for name, data in figs:
                    self.put_raw(name, data)

            found = {}
            not_found = []
            for k, v in collector.aliases.items():
                if [item for item in v if item != k]:
                    if shorter := find_cannonical(k, v):
                        found[k] = shorter
                    else:
                        not_found.append((k, v))

            if logo := module_conf.get("logo", None):
                self.put_raw("logo.png", Path(logo).read_bytes())
            self.metadata = {
                "version": version,
                "logo": "logo.png",
                "aliases": found,
                "module": root,
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
