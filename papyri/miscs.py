"""
Misc helper functions
"""

import io
import sys
import ast

from rich.progress import Progress

from contextlib import redirect_stdout, redirect_stderr, contextmanager


@contextmanager
def capture_displayhook(acc):
    def dh(value):
        acc.append(value)

    old_dh = sys.displayhook
    try:
        sys.displayhook = dh
        yield
    finally:
        sys.displayhook = old_dh


class DummyP(Progress):
    """
    Rich progress bar can screw up ipdb, so it can be useful to have a dummy
    replacement
    """

    def add_task(*args, **kwargs):
        pass

    def advance(*args, **kwargs):
        pass

    def update(*args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass


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
        assert (len(self.fig_man())) == 0, f"init fail in {len(self.fig_man())}"

    def __exit__(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        plt.close("all")
        assert (len(self.fig_man())) == 0, f"init fail in {len(self.fig_man())}"

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

    def _exec(self, text, ns, name):
        """
        A variant of exec that can run multi line,
        and capture sys_displayhook
        """
        module = ast.parse(text)
        if not module.body:  # this can happen if we execute purely a comment.
            return None
        try:
            *nodes, interactive_node = module.body
        except Exception as e:
            raise type(e)(f"{module.body} {text}")
        exec(compile(ast.Module(nodes, []), name, "exec"), ns)
        acc = []
        with capture_displayhook(acc):
            exec(compile(ast.Interactive([interactive_node]), name, "single"), ns)
        if len(acc) == 1:
            return acc[0]
        else:
            return None

    def exec(self, text, *, name='<papyri>'):
        from matplotlib import _pylab_helpers, cbook
        from matplotlib.backend_bases import FigureManagerBase

        stdout = io.StringIO()
        stderr = io.StringIO()
        with cbook._setattr_cm(FigureManagerBase, show=lambda self: None):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                res = self._exec(text, self.ns, name)

        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()

        stdout.seek(0)
        stderr.seek(0)
        return res, fig_managers, stdout.read(), stderr.read()
