import inspect
import json
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from os.path import expanduser
from pathlib import Path
from textwrap import dedent

# from numpydoc.docscrape import NumpyDocString
from types import FunctionType, ModuleType

import jedi
import numpy as np
from numpy import array2string
from pygments.lexers import PythonLexer
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    ProgressColumn,
    TaskID,
    Text,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from there import print
from velin.examples_section_utils import InOut, splitblank, splitcode
from velin.ref import NumpyDocString

from . import utils
from .config import base_dir, cache_dir
from .utils import progress


@lru_cache()
def keepref(ref):
    """
    Filter to rim out common reference that we usually do not want to keep
    around in examples; typically most of the builtins, and things we can't
    import.
    """
    if ref.startswith(("builtins.", "__main__")):
        return False
    try:
        __import__(ref)
        return False
    except Exception:
        pass
    return True


def dedent_but_first(text):
    """
    simple version of `inspect.cleandoc` that does not trim empty lines
    """
    a, *b = text.split("\n")
    return dedent(a) + "\n" + dedent("\n".join(b))


def pos_to_nl(script: str, pos: int) -> (int, int):
    """
    Convert pigments position to Jedi col/line
    """
    rest = pos
    ln = 0
    for line in script.splitlines():
        if len(line) < rest:
            rest -= len(line) + 1
            ln += 1
        else:
            return ln, rest


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
                    raise
                    pass
                break
        except IndexError:
            raise
            ref = ""
        yield text, ref
    warnings.simplefilter("default", UserWarning)


counter = 0


def get_example_data(doc, infer=True, obj=None, exec_=True):
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
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import FigureManagerBase
    from matplotlib import cbook, _pylab_helpers

    acc = ""
    ns = {"np": np, "plt": plt, obj.__name__: obj}
    for b in blocks:
        for item in b:
            if isinstance(item, InOut):
                script = "\n".join(item.in_)
                fig = None
                if exec_:
                    try:
                        with cbook._setattr_cm(
                            FigureManagerBase, show=lambda self: None
                        ):
                            exec(script, ns)
                        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
                        if fig_managers:
                            print("figs !", fig_managers)
                            global counter
                            counter += 1
                            figman = next(iter(fig_managers))
                            from pathlib import Path
                            import os.path

                            p = (
                                Path(os.path.expanduser("~/.papyri"))
                                / f"fig-{obj.__name__}-{counter}.png"
                            )
                            figman.canvas.figure.savefig(
                                p, dpi=300, bbox_inches="tight"
                            )
                            plt.close("all")
                            fig = str(p.absolute())

                    except:
                        pass
                entries = list(parse_script(script, ns=ns, infer=infer, prev=acc))
                acc += "\n" + script
                edata.append(["code", (entries, "\n".join(item.out))])
                if fig:
                    edata.append(["fig", fig])
            else:
                edata.append(["text", "\n".join(item.out)])
    return edata


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
    Gen().do_one_mod(names, infer, exec_)


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

    def render(self, task: "Task") -> Text:
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
            raise RuntimeError(f"error visiting {'.'.join(self.stack)}")
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


class Gen:
    def __init__(self):
        self.cache_dir = cache_dir

    def clean(self, root):
        bundle = self.cache_dir / root
        for _, path in progress(
            bundle.glob("*.json"), description="cleaning previous bundle"
        ):
            path.unlink()
        bundle.mkdir(exist_ok=True)

    def put(self, root, path, data):
        with (self.cache_dir / root / f"{path}.json").open("w") as f:
            f.write(data)

    def do_one_mod(self, names, infer, exec_):

        p = lambda: Progress(
            TextColumn("[progress.description]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "[progress.completed]{task.completed} / {task.total}",
            TimeElapsedColumn(),
        )

        modules = []
        for name in names:
            x, *r = name.split(".")
            n0 = __import__(name)
            for sub in r:
                n0 = getattr(n0, sub)
            modules.append(n0)

        version = getattr(modules[0], "__version__", "???")

        root = names[0].split(".")[0]
        nvisited_items = {}
        task = None

        self.clean(root)

        collected = Collector(n0).items()

        for qa, item in collected.items():
            if (nqa := full_qual(item)) != qa:
                print("after import qa differs : {qa} -> {nqa}")
                if (other := collected[nqa]) == item:
                    print("present twice")
                    del collected[nqa]
                else:
                    print("differs: {item} != {other}")

        with p() as p2:
            bundle = self.cache_dir / root
            taskp = p2.add_task(description="parsing", total=len(collected))
            t1 = timer(p2, taskp)
            if infer:
                taski = p2.add_task(
                    description="Running type inference in examples",
                    total=len(collected),
                )
                t2 = timer(p2, taski)
            for qa, a in collected.items():
                short_description = (qa[:19] + "..") if len(qa) > 21 else qa
                p2.update(taskp, description=short_description.ljust(17))
                ddd = a.__doc__
                if ddd is None:
                    p2.advance(taskp)
                    if infer:
                        p2.advance(taski)
                    continue

                # progress.console.print(qa)
                with t1():
                    try:
                        ndoc = NumpyDocString(dedent_but_first(ddd))
                    except:
                        p2.console.print("Unexpected error parsing", a)
                        p2.advance(taskp)
                        if infer:
                            p2.advance(taski)
                        continue
                p2.advance(taskp)

                if not ndoc["Signature"]:
                    sig = None
                    try:
                        sig = str(inspect.signature(a))
                    except (ValueError, TypeError):
                        pass
                    if sig:
                        ndoc["Signature"] = qa.split(".")[-1] + sig

                new_see_also = ndoc["See Also"]
                refs = []
                if new_see_also:
                    for line in new_see_also:
                        rt, desc = line
                        for ref, type_ in rt:
                            refs.append(ref)

                if getattr(nvisited_items, qa, None):
                    raise ValueError(f"{qa} already visited")
                try:
                    if infer:
                        with t2():
                            ndoc.edata = get_example_data(
                                ndoc, infer, obj=a, exec_=exec_
                            )
                    else:
                        ndoc.edata = get_example_data(ndoc, infer, obj=a, exec_=exec_)
                except Exception:
                    ndoc.edata = []
                    print("Error getting example date in ", qa)

                ndoc.refs = list(
                    {
                        u[1]
                        for t_, sect in ndoc.edata
                        if t_ == "code"
                        for u in sect[0]
                        if u[1]
                    }
                )
                ndoc.refs.extend(refs)
                ndoc.refs = [normalise_ref(r) for r in sorted(set(ndoc.refs))]
                if infer:
                    p2.advance(taski)
                self.put(root, qa, json.dumps(ndoc.to_json(), indent=2))
                nvisited_items[qa] = ndoc
            self.put(root, "__papyri__", json.dumps({"version": version}))
