import re

from IPython.core.magic import Magics, line_magic, magics_class


# The class MUST call this class decorator at creation time
@magics_class
class Papyri(Magics):
    @line_magic
    def pinfo(self, parameter_s="", namespaces=None):
        """Provide detailed information about an object.

        '%pinfo object' is just a synonym for object? or ?object."""

        from papyri.browser import main
        from papyri.gen import full_qual

        pinfo, qmark1, oname, qmark2 = re.match(
            r"(pinfo )?(\?*)(.*?)(\??$)", parameter_s
        ).groups()

        if _ := main(parameter_s):
            return
        else:
            parts_1 = oname.split(".")
            other = []
            name, *other = parts_1

            obj = self.shell.user_ns.get(name, None)
            for o in other:
                obj = getattr(obj, o)
            if obj is not None:
                qa = full_qual(obj)
                if _ := main(qa):
                    return

        # print 'pinfo par: <%s>' % parameter_s  # dbg
        # detail_level: 0 -> obj? , 1 -> obj??
        detail_level = 0
        # We need to detect if we got called as 'pinfo pinfo foo', which can
        # happen if the user types 'pinfo foo?' at the cmd line.
        if pinfo or qmark1 or qmark2:
            detail_level = 1
        if "*" in oname:
            self.psearch(oname)
        else:
            self.shell._inspect(
                "pinfo", oname, detail_level=detail_level, namespaces=namespaces
            )


# In order to actually use these magics, you must register them with a
# running IPython.


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(Papyri)
