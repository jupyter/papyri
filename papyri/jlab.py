from IPython.core.magic import Magics, magics_class, line_magic
from IPython.display import IFrame
import sys

# The class MUST call this class decorator at creation time
@magics_class
class Papyri(Magics):
    def newhelp(self, obj):
        mod = obj.__module__
        root = mod.split(".")[0]
        fq = mod + "." + obj.__name__

        version = sys.modules[root].__version__
        return IFrame(
            f"http://127.0.0.1:1234/p/{root}/{version}/api/{fq}", "100%", "100%"
        )

    @line_magic
    def papyri(self, line):
        path0, *path = line.strip().split(".")
        obj = self.shell.user_ns[path0]
        for p in path:
            obj = getattr(obj, p)
        return self.newhelp(obj)


get_ipython().register_magics(Papyri)
