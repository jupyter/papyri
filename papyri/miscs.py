"""
Misc helper functions
"""

import io

from rich.progress import Progress


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

    def exec(self, text):
        from matplotlib import _pylab_helpers, cbook
        from matplotlib.backend_bases import FigureManagerBase

        with cbook._setattr_cm(FigureManagerBase, show=lambda self: None):
            res = exec(text, self.ns)

        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()

        return res, fig_managers
