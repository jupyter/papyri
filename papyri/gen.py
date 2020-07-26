import inspect
import json
import sys
import time
from contextlib import contextmanager
from os.path import expanduser
from pathlib import Path
from textwrap import dedent
# from numpydoc.docscrape import NumpyDocString
from types import ModuleType

import jedi
import matplotlib
import matplotlib.pyplot
import numpy as np
import numpy.core.numeric
import scipy
import scipy.special
import sklearn
from pygments.lexers import PythonLexer
from rich.progress import (BarColumn, DownloadColumn, Progress, ProgressColumn,
                           TaskID, Text, TextColumn, TimeRemainingColumn,
                           TransferSpeedColumn)
from there import print
from velin.examples_section_utils import InOut, splitblank, splitcode
from velin.ref import NumpyDocString

from . import utils
from .config import base_dir, cache_dir


def dedent_but_first(text):
    a, *b = text.split("\n")
    return dedent(a) + "\n" + dedent("\n".join(b))


def pos_to_nl(script, pos):
    rest = pos
    ln = 0
    for line in script.splitlines():
        if len(line) < rest:
            rest -= len(line) + 1
            ln += 1
        else:
            return ln, rest


P = PythonLexer()


def parse_script(script, ns=None, infer=None):

    jeds = []
    if ns:
        jeds.append(jedi.Interpreter(script, namespaces=[ns]))
    jeds.append(jedi.Script(script))

    for index, type_, text in P.get_tokens_unprocessed(script):
        a, b = pos_to_nl(script, index)
        try:
            ref = None
            for jed in jeds:
                try:
                    if infer:
                        ref = jed.infer(a + 1, b)[0].full_name
                    else:
                        ref = ""
                except (AttributeError, TypeError, Exception):
                    pass
                break
        except IndexError:
            ref = ""
        yield index, type_, text, ref


def get_example_data(doc, infer=True):
    blocks = list(map(splitcode, splitblank(doc["Examples"])))
    edata = []
    for b in blocks:
        for item in b:
            if isinstance(item, InOut):
                script = "\n".join(item.in_)
                entries = list(parse_script(script, ns={"np": np}, infer=infer))
                edata.append(["code", (entries, "\n".join(item.out))])

            else:
                edata.append(["text", "\n".join(item.out)])
    return edata


def main(names, infer):
    for name in names:
        do_one_mod(name, infer)


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


def do_one_mod(name, infer):
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
            return Text(
                str(ctime_delta), style="progress.remaining", overflow="ellipsis"
            )

    p = lambda: Progress(
        TextColumn("[progress.description]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "[progress.completed]{task.completed} / {task.total}",
        TimeElapsedColumn(),
    )

    x, *r = name.split(".")
    n0 = __import__(name)
    for sub in r:
        n0 = getattr(n0, sub)
    modules = [n0]

    root = name.split(".")[0]
    collected = {}
    nvisited_items = {}
    task = None
    with p() as progress:
        for mod in modules:
            if not mod.__name__.startswith(name):
                # progress.console.print("skiping module not submodule of", mod, mod.__name__)
                continue
                # pass

            # progress.console.print(mod.__name__)
            if task is not None:
                progress.remove_task(task)
            task = progress.add_task(
                description="collecting... " + mod.__name__, total=len(dir(mod))
            )
            for n in dir(mod):
                progress.advance(task)
                try:
                    a = getattr(mod, n)
                except Exception:
                    continue
                if isinstance(a, ModuleType):
                    if a not in modules:
                        modules.append(a)
                    continue
                if getattr(a, "__module__", None) is None:
                    continue
                if isinstance(lqa := getattr(a, "__qualname__", None), str):
                    qa = a.__module__ + "." + lqa
                else:
                    qa = a.__module__ + "." + n
                if not qa.startswith(root):
                    continue
                if not isinstance(ddd := getattr(a, "__doc__", None), str):
                    continue
                collected[qa] = a

    # with progress:
    with p() as p2:
        taskp = p2.add_task(description="parsing", total=len(collected))
        t1 = timer(p2, taskp)
        if infer:
            taski = p2.add_task(description="infering examples", total=len(collected))
            t2 = timer(p2, taski)
        for qa, a in collected.items():
            sd = (qa[:19] + "..") if len(qa) > 21 else qa
            p2.update(taskp, description=sd.ljust(17))
            ddd = a.__doc__

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
            if infer:
                with t2():
                    ndoc.edata = get_example_data(ndoc, infer)
            else:
                ndoc.edata = get_example_data(ndoc, infer)

            ndoc.refs = list(
                {
                    u[3]
                    for t_, sect in ndoc.edata
                    if t_ == "code"
                    for u in sect[0]
                    if u[3]
                }
            )
            ndoc.refs.extend(refs)
            ndoc.refs = list(sorted(set(ndoc.refs)))
            if infer:
                p2.advance(taski)
            ndoc.backrefs = []

            with (cache_dir / f"{qa}.json").open("w") as f:
                f.write(json.dumps(ndoc.to_json()))
            nvisited_items[qa] = ndoc


if __name__ == "__main__":
    main()
