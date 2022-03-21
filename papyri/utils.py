import time
from datetime import timedelta
from textwrap import dedent
from typing import Tuple

from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.text import Text
from types import ModuleType


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


class TimeElapsedColumn(ProgressColumn):

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(self, *args, **kwargs):
        self.avg = None
        super().__init__(*args, **kwargs)

    def render(self, task: "Task"):
        # task.completed
        # task.total
        elapsed = task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        elapsed_delta = timedelta(seconds=int(elapsed))
        if task.time_remaining is not None:
            if self.avg is None:
                self.avg = elapsed_delta + timedelta(seconds=int(task.time_remaining))
            else:
                self.avg = (
                    99 * self.avg
                    + elapsed_delta
                    + timedelta(seconds=int(task.time_remaining))
                ) / 100
            finish_delta = str(self.avg).split(".")[0]
        else:
            finish_delta = "--:--:--"
        return Text(
            str(elapsed_delta) + "/" + str(finish_delta), style="progress.elapsed"
        )


def dummy_progress(
    iterable,
    *,
    description="Progress",
    transient=True,
):
    items = list(iterable)
    it = iter(items)
    now = time.monotonic()

    def gen():
        try:
            c = 0
            while True:
                yield None, next(it)
                c += 1
        except StopIteration:
            if transient:
                deltat = time.monotonic() - now
                print(
                    description,
                    f"Done {c} items in {deltat:.2f} seconds ({int(c/deltat)} item/s)",
                )
            return
        except BaseException:
            raise

    return gen()


def progress(iterable, *, description="Progress", transient=True):
    items = list(iterable)
    p = Progress(
        TextColumn("[progress.description]{task.description:15}", justify="left"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.completed}/{task.total}",
        TimeElapsedColumn(),
        transient=transient,
    )
    p.start()
    task = p.add_task(description, total=len(items), ee=0)
    it = iter(items)
    now = time.monotonic()

    def gen():
        try:
            c = 0
            while True:
                p.update(task, ee=time.monotonic() - now)
                p.advance(task)
                yield p, next(it)
                c += 1
        except StopIteration:
            p.stop()
            if transient:
                deltat = time.monotonic() - now
                print(
                    description,
                    f"Done {c} items in {deltat:.2f} seconds ({int(c/deltat)} item/s)",
                )
            return
        except BaseException:
            p.stop()
            raise

    return gen()


def dedent_but_first(text):
    """
    simple version of `inspect.cleandoc` that does not trim empty lines
    """
    assert isinstance(text, str), (text, type(text))
    a, *b = text.split("\n")
    return dedent(a) + "\n" + dedent("\n".join(b))


def pos_to_nl(script: str, pos: int) -> Tuple[int, int]:
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
    raise RuntimeError
