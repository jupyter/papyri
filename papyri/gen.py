from __future__ import annotations
import inspect
import io
import json
import time
from contextlib import contextmanager, nullcontext
from functools import lru_cache

# from numpydoc.docscrape import NumpyDocString
from types import FunctionType, ModuleType
from typing import List, Dict, Any, Tuple

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

from .config import cache_dir
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
    edata = []
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
                if exec_:
                    try:
                        with cbook._setattr_cm(
                            FigureManagerBase, show=lambda self: None
                        ):
                            exec(script, ns)
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

                    except Exception:
                        raise
                entries = list(parse_script(script, ns=ns, infer=infer, prev=acc))
                acc += "\n" + script
                edata.append(["code", (entries, "\n".join(item.out))])
                if figname:
                    edata.append(["fig", figname])
            else:
                edata.append(["text", "\n".join(item.out)])
    return edata, figs


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


class Collector:
    def __init__(self, root):
        assert isinstance(root, ModuleType), root
        self.root = root
        self.obj = dict()
        self.stack = [self.root.__name__]

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
        try:
            qa = full_qual(obj)
        except Exception as e:
            raise RuntimeError(f"error visiting {'.'.join(self.stack)}") from e
        if not qa:
            return
        if not qa.startswith(self.root.__name__):
            return
        if obj in self.obj.values():
            return
        if (qa in self.obj) and self.obj[qa] != obj:
            pass
        self.obj[qa] = obj

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

    __slots__ = (
        "_content",
        "example_section_data",
        "refs",
        "ordered_sections",
        "item_file",
        "item_line",
        "item_type",
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
        ]

    def __init__(self):
        self._content = None
        self.example_section_data = None
        self.refs = None
        self.ordered_sections = None
        self.item_file = None
        self.item_line = None
        self.item_type = None

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

        res = {k: getattr(self, k) for k in self.slots()}

        return res

    @classmethod
    def from_json(cls, obj):
        nds = cls()
        for k in obj:
            setattr(nds, k, obj.get(k))
        if nds._content is None:
            nds._content = {}
        nds._content["Parameters"] = [
            Parameter(a, b, c) for (a, b, c) in nds._content.get("Parameters", [])
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
            if it not in nds._content:
                nds._content[it] = []
        for it in ("index",):
            if it not in nds._content:
                nds._content[it] = {}
        return nds


class Gen:
    def __init__(self):
        self.cache_dir = cache_dir
        self.data = {}
        self.bdata = {}

    def clean(self, where: Path):
        where = where / self.root
        for _, path in progress(
            where.glob("*.json"), description="cleaning previous bundle"
        ):
            path.unlink()

    def write(self, where: Path):
        assert self.root is not None
        (where / self.root).mkdir(exist_ok=True)
        for k, v in self.data.items():
            with (where / self.root / k).open("w") as f:
                f.write(v)

        for k, v in self.bdata.items():
            with (where / self.root / k).open("wb") as f:
                f.write(v)

    def put(self, root, path, data):
        self.data[path + ".json"] = data

    def put_raw(self, root, path, data):
        self.bdata[path] = data

    def do_one_item(
        self, target_item, ndoc, infer: bool, exec_, qa
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
                for ref, type_ in rt:
                    refs.append(ref)

        try:
            ndoc.edata, figs = get_example_data(
                ndoc, infer, obj=target_item, exec_=exec_, qa=qa
            )
            ndoc.figs = figs
        except Exception as e:
            ndoc.edata = []
            print("Error getting example data in ", qa)
            raise ValueError("Error getting example data in ", qa) from e

        ndoc.refs = list(
            {u[1] for t_, sect in ndoc.edata if t_ == "code" for u in sect[0] if u[1]}
        )
        ndoc.refs.extend(refs)
        ndoc.refs = [normalise_ref(r) for r in sorted(set(ndoc.refs))]
        figs = ndoc.figs
        del ndoc.figs

        blob.example_section_data = ndoc.edata
        blob.ordered_sections = ndoc.ordered_sections
        blob.refs = ndoc.refs
        blob.item_file = item_file
        blob.item_line = item_line
        blob.item_type = item_type

        # del ndoc.edata
        # del ndoc.refs
        # turn the numpydoc thing into a docblob
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

        print("Configuration:", conf)
        self.root = root
        self.version = version

        # clean out previous doc bundle

        collected: Dict[str, Any] = Collector(n0).items()

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
                if item_docstring is None:
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
                else:
                    print("will run", qa)

                try:
                    doc_blob, figs = self.do_one_item(target_item, ndoc, infer, ex, qa)
                except Exception:
                    if module_conf.get("exec_failure", None) == "fallback":
                        print("Re-analysing ", qa, "without execution")
                        # debug:
                        doc_blob, figs = self.do_one_item(
                            target_item, ndoc, infer, False, qa
                        )

                self.put(root, qa, json.dumps(doc_blob.to_json(), indent=2))
                for name, data in figs:
                    self.put_raw(root, name, data)

            if logo := module_conf.get("logo", None):
                self.put_raw(root, f"{root}-logo.png", Path(logo).read_bytes())
                self.put(
                    root,
                    "__papyri__",
                    json.dumps({"version": version, "logo": f"{root}-logo.png"}),
                )
            else:
                self.put(root, "__papyri__", json.dumps({"version": version}))
