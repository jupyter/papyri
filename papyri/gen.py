from __future__ import annotations
import inspect
import io
import json
import time
from contextlib import contextmanager, nullcontext
from collections import defaultdict
from functools import lru_cache

# from numpydoc.docscrape import NumpyDocString
from types import FunctionType, ModuleType
from typing import List, Dict, Any, Tuple, Optional

import jedi
from pygments.lexers import PythonLexer
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Text,
    TextColumn,
)
from there import print
from velin.examples_section_utils import InOut, splitblank, splitcode
from .vref import NumpyDocString
from numpydoc.docscrape import Parameter

from .utils import pos_to_nl, dedent_but_first, progress

from pathlib import Path


def parse_script(script, ns=None, infer=None, prev=""):
    """
    Parse a script into tokens and use Jedi to infer the fully qualified names
    of each token.

    Parameters
    ----------
    script : str
        the script to tokenize and infer types on
    ns : dict
        extra namesapce to use with jedi's Interpreter.
    infer : bool
        whether to run jedi type inference that can be quite time consuming.
    prev : str
        previsou lines that lead to this.

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
                except (AttributeError, TypeError, Exception):
                    # raise
                    failed = "(jedi failed inference)"
                    print("failed inference on ", script, ns, jed, col_n, line_n + 1)
                break
        except IndexError:
            raise
            ref = ""
        yield text + failed, ref
    warnings.simplefilter("default", UserWarning)


counter = 0


def get_example_data(doc, infer=True, obj=None, exec_=True, qa=None):
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
    blocks = list(map(splitcode, splitblank(doc["Examples"])))
    example_section_data = []
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import _pylab_helpers, cbook
    from matplotlib.backend_bases import FigureManagerBase

    matplotlib.use("agg")

    acc = ""
    import numpy as np

    ns = {"np": np, "plt": plt, obj.__name__: obj}
    figs = []
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
                if exec_:
                    try:
                        with cbook._setattr_cm(
                            FigureManagerBase, show=lambda self: None
                        ):
                            exec(script, ns)
                            ce_status = "execed"
                        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
                        if fig_managers:
                            global counter
                            counter += 1
                            figman = next(iter(fig_managers))

                            buf = io.BytesIO()
                            if not qa:
                                qa = obj.__name__
                            figname = f"fig-{qa}-{counter}.png"
                            figman.canvas.figure.savefig(
                                buf, dpi=300, bbox_inches="tight"
                            )
                            plt.close("all")
                            buf.seek(0)
                            figs.append((figname, buf.read()))

                    except Exception as e:
                        print("exception executing...")
                        pass
                        # import traceback
                        # traceback.print_exc()
                        # print(script)
                        # print(e)
                        # raise
                entries = list(parse_script(script, ns=ns, infer=infer, prev=acc))
                acc += "\n" + script
                example_section_data.append(
                    ["code", (entries, "\n".join(item.out), ce_status)]
                )
                if figname:
                    example_section_data.append(["fig", figname])
            else:
                assert isinstance(item.out, list)
                example_section_data.append(["text", "\n".join(item.out)])

    return example_section_data, figs


@lru_cache()
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

    conffile = Path("papyri.toml")
    if conffile.exists():
        conf = toml.loads(conffile.read_text())
    else:
        conf = {}

    g = Gen()
    g.do_one_mod(names, infer, exec_, conf)
    p = Path(".") / (g.root + "_" + g.version)
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

    def render(self, task) -> Text:
        """Show time remaining."""
        from datetime import timedelta

        ctime = task.fields.get("ctime", None)
        if ctime is None:
            return Text("-:--:--", style="progress.remaining")
        ctime_delta = timedelta(seconds=int(ctime))
        return Text(str(ctime_delta), style="progress.remaining", overflow="ellipsis")


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
        self.root = root
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
        if not qa.startswith(self.root.__name__):
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


class DocBlob:
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

    _content: dict
    refs: list
    ordered_sections: list
    item_file: Optional[str]
    item_line: Optional[int]
    item_type: Optional[str]
    aliases: dict
    example_section_data: list

    __slots__ = (
        "_content",
        "example_section_data",
        "refs",
        "ordered_sections",
        "item_file",
        "item_line",
        "item_type",
        "aliases",
        "logo",
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
            "aliases",
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

    def to_json(self):

        res = {k: getattr(self, k, "") for k in self.slots()}

        return res

    @classmethod
    def from_json(cls, obj):
        new_doc_blob = cls()
        for k, v in obj.items():
            setattr(new_doc_blob, k, v)
        if new_doc_blob._content is None:
            new_doc_blob._content = {}

        new_doc_blob._content["Parameters"] = [
            Parameter(a, b, c)
            for (a, b, c) in new_doc_blob._content.get("Parameters", [])
        ]

        for it in (
            "Returns",
            "Yields",
            "Extended Summary",
            "Receives",
            "Other Parameters",
            "Raises",
            "Warns",
            "Warnings",
            "See Also",
            "Notes",
            "References",
            "Examples",
            "Attributes",
            "Methods",
        ):
            if it not in new_doc_blob._content:
                new_doc_blob._content[it] = []
        for it in ("index",):
            if it not in new_doc_blob._content:
                new_doc_blob._content[it] = {}
        return new_doc_blob


class Gen:
    def __init__(self):
        self.data = {}
        self.bdata = {}
        self.metadata = {}

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
        self, target_item: Any, ndoc, infer: bool, exec_: bool, qa: str
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
        blob = DocBlob()
        import copy

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
                ndoc, infer, obj=target_item, exec_=exec_, qa=qa
            )
            ndoc.figs = figs
        except Exception as e:
            ndoc.example_section_data = []
            print("Error getting example data in ", qa)
            raise ValueError("Error getting example data in ", qa) from e

        ndoc.refs = list(
            {
                u[1]
                for t_, sect in ndoc.example_section_data
                if t_ == "code"
                for u in sect[0]
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

        print("Configuration:", json.dumps(module_conf, indent=2))
        self.root = root
        self.version = version

        # clean out previous doc bundle
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

        # p = nullcontext
        with p() as p2:

            # just nice display of progression.
            taskp = p2.add_task(description="parsing", total=len(collected))
            t1 = timer(p2, taskp)

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

                # progress.console.print(qa)
                t1 = nullcontext
                with t1():
                    try:
                        ndoc = NumpyDocString(dedent_but_first(item_docstring))
                    except Exception:
                        p2.console.print(
                            "Unexpected error parsing",
                            target_item,
                            target_item.__name__,
                        )
                        if isinstance(target_item, ModuleType):
                            ndoc = NumpyDocString(
                                f"Was not able to parse docstring for {qa}"
                            )
                        else:
                            continue
                execute_exclude_patterns = module_conf.get(
                    "execute_exclude_patterns", None
                )
                ex = exec_
                if execute_exclude_patterns and exec_:
                    for pat in execute_exclude_patterns:
                        if qa.startswith(pat):
                            print("will not execute", qa)
                            ex = False
                            break
                # else:
                #    print("will run", qa)

                try:
                    doc_blob, figs = self.do_one_item(target_item, ndoc, infer, ex, qa)
                except Exception:
                    raise
                    if module_conf.get("exec_failure", None) == "fallback":
                        print("Re-analysing ", qa, "without execution")
                        # debug:
                        doc_blob, figs = self.do_one_item(
                            target_item, ndoc, infer, False, qa
                        )
                doc_blob.aliases = collector.aliases[qa]
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
                self.put_raw(f"logo.png", Path(logo).read_bytes())
            self.metadata = {
                "version": version,
                "logo": f"logo.png",
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
