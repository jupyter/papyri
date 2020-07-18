import ast
import json
import sys
import difflib
from textwrap import indent
from there import print

import numpydoc.docscrape as nds

from .examples_section_utils import reformat_example_lines

import ast


class NodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.items = []

    def visit_FunctionDef(self, node):
        # we can collect function args, and _could_ check the Parameter section.
        # for nn in node.args.args:
        #    nnn = nn.arg
        # for k in [l for l in dir(node) if not l.startswith('_')]:
        #    sub = getattr(node, k)
        #    print(k, sub)
        #    for u in [l for l in dir(sub) if not l.startswith('_')]:
        #        ss = getattr(sub, u)
        #        print('    ', u, ss)

        # node.args.kwarg # single object
        # node.args.vararg # object
        # node.args.ks_defaults #list
        # print(node.returns)
        # node.args.defaults +

        meta = [
            arg.arg
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        ]
        #    print(arg.arg)
        #    for k in [l for l in dir(arg) if not l.startswith('_')]:
        #        sub = getattr(arg, k)
        #        print('    ', k, sub)

        self.items.append((node, meta))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        # we can collect function args, and _could_ check the Parameter section.
        # for nn in node.args.args:
        #    nnn = nn.arg

        # self.items.append((node, None))
        self.generic_visit(node)


BLACK_REFORMAT = True


class NumpyDocString(nds.NumpyDocString):
    """
    subclass a littel bit more lenient on parsing
    """

    aliases = {
        "Parameters": (
            "options",
            "parameter",
            "parameters",
            "paramters",
            "parmeters",
            "paramerters",
            "arguments",
        ),
        "Yields": ("signals",),
    }

    def normalize(self):
        """
        Apply a bunch of heuristic that try to normalise the data.
        """
        if (params := self["Parameters"]) :
            for i, p in enumerate(params):
                if not p.type and ":" in p.name:
                    if p.name.startswith(".."):
                        continue
                    name, type_ = [_.strip() for _ in p.name.split(":", maxsplit=1)]
                    params[i] = nds.Parameter(name, type_, p[2])

    def parse_examples(self, lines, indent=4):
        # this is bad practice be we do want normalisation here for now
        # to  check that parse->format->parse is idempotent.
        # this can be done if we had a separate "normalize" step.
        global BLACK_REFORMAT
        if BLACK_REFORMAT:
            try:
                lines = reformat_example_lines(lines, indent=indent)
            except Exception:
                print("black failed")
                print("\n".join(lines))
                raise

        return lines

    def to_json(self):
        return {
            "_parsed_data": self._parsed_data,
            "edata": self.edata,
            "refs": self.refs,
            "backref": self.backrefs,
        }

    def __init__(self, *args, **kwargs):
        self.ordered_sections = []
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in ["Extended Summary", "Summary"]:
            value = [d.rstrip() for d in value]

        if key in ("Examples"):
            value = self.parse_examples(value)
        super().__setitem__(key, value)
        assert key not in self.ordered_sections
        self.ordered_sections.append(key)

    def _guess_header(self, header):
        if header in self.sections:
            return header
        # handle missing trailing `s`, and trailing `:`
        for s in self.sections:
            if s.lower().startswith(header.rstrip(":").lower()):
                return s
        for k, v in self.aliases.items():
            if header.lower() in v:
                return k
        raise ValueError("Cound not find match for section:", header)

    def _read_sections(self):
        for name, data in super()._read_sections():
            name = self._guess_header(name)
            yield name, data

    def _parse_param_list(self, *args, **kwargs):
        """
        Normalize parameters
        """
        parms = super()._parse_param_list(*args, **kwargs)
        out = []
        for name, type_, desc in parms:
            out.append(
                nds.Parameter(name.strip(), type_.strip(), [d.rstrip() for d in desc])
            )
        return out


def w(orig):
    """Util function that shows whitespace as `·`."""
    ll = []
    for l in orig:
        if l[0] in "+-":
            # ll.append(l.replace(' ', '⎵'))
            ll.append(l.replace(" ", "·"))

        else:
            ll.append(l)
    lll = []
    for l in ll:
        if l.endswith("\n"):
            lll.append(l[:-1])
        else:
            lll.append(l[:])
    return lll


class Conf:
    """
    
    Here are some of the config options

    - F100: Items spacing in Return/Raise/Yield/Parameters/See Also
        - A(auto) - spaces if some sections have blank lines. (default)
        - B(compact) - no spaces between items, ever.
        - C(spaced)  - spaces between items, always

    """

    defaults = {"F100": "A"}

    def __init__(self, conf):
        self._conf = conf

    def __getattr__(self, key):
        k = key[:-1]
        v = key[-1]
        return self._conf.get(k, self.defaults[k]) == v


class SectionFormatter:
    """
    Render section of the docs, based on some configuration. Not having
    configuration is great, but everybody has their pet peevs, and I'm hpping we
    can progressively convince them to adopt a standard.
    """

    def __init__(self, *, conf):
        self.config = Conf(conf)

    @classmethod
    def format_Signature(self, s, compact):
        return s + "\n"

    @classmethod
    def format_Summary(self, s, compact):
        if len(s) == 1 and not s[0].strip():
            return ""
        return "\n".join(s) + "\n"

    @classmethod
    def format_Extended_Summary(self, es, compact):
        return "\n".join(es) + "\n"

    def _format_ps(self, name, ps, compact):
        force_compact = self.config.F100B
        res, try_other = self._format_ps_pref(name, ps, compact=True)
        if (not try_other and self.config.F100A) or self.config.F100B:
            return res
        return self._format_ps_pref(name, ps, compact=False)[0]

    def _format_ps_pref(self, name, ps, *, compact):
        try_other = False
        out = name + "\n"
        out += "-" * len(name) + "\n"
        for i, p in enumerate(ps):
            if (not compact) and i:
                out += "\n"
            if p.type:
                out += f"""{p.name.strip()} : {p.type.strip()}\n"""
            else:
                out += f"""{p.name.strip()}\n"""
            if p.desc:
                if any([l.strip() == "" for l in p.desc]):
                    try_other = True

                out += indent("\n".join(p.desc), "    ")
                out += "\n"
        return out, try_other

    def format_Parameters(self, ps, compact):
        return self._format_ps("Parameters", ps, compact)

    def format_Methods(self, ps, compact):
        return self._format_ps("Methods", ps, compact)

    def format_Other_Parameters(self, ps, compact):
        return self._format_ps("Other Parameters", ps, compact)

    @classmethod
    def format_See_Also(cls, sas, compact):

        res = cls.format_See_Also_impl(sas, True, force_compact=compact)
        if res is not None:
            return res
        return cls.format_See_Also_impl(sas, False, force_compact=compact)

    @classmethod
    def format_See_Also_impl(cls, sas, compact, force_compact):
        out = "See Also\n"
        out += "--------\n"

        for a, b in sas:
            if b:
                desc = b[0]
            else:
                desc = None
            if len(b) > 1:
                rest_desc = b[1:]
            else:
                rest_desc = []
            _first = True
            for ref, type_ in a:
                if not _first:
                    out += ", "
                if type_ is not None:
                    out += f":{type_}:`{ref}`"
                else:
                    out += f"{ref}"
                _first = False

            if desc:
                if len(a) > 1 or (not compact):
                    out += f"\n    {desc}"
                else:
                    attempt = f" : {desc}"
                    if len(out.splitlines()[-1] + attempt) > 75 and not force_compact:
                        return None
                    out += attempt
            for rd in rest_desc:
                out += "\n    " + rd
            out += "\n"
        return out

    @classmethod
    def format_References(cls, lines, compact):
        out = "References\n"
        out += "----------\n"
        out += "\n".join(lines)
        out += "\n"
        return out

    @classmethod
    def format_Notes(cls, lines, compact):
        out = "Notes\n"
        out += "-----\n"
        out += "\n".join(lines)
        out += "\n"
        return out

    @classmethod
    def format_Examples(cls, lines, compact):
        out = "Examples\n"
        out += "--------\n"
        out += "\n".join(lines)
        out += "\n"
        return out

    @classmethod
    def format_Warnings(cls, lines, compact):
        out = "Warnings\n"
        out += "--------\n"
        out += "\n".join(lines)
        out += "\n"
        return out

    @classmethod
    def format_Warns(cls, ps, compact):
        return cls.format_RRY("Warns", ps)

    @classmethod
    def format_Raises(cls, ps, compact):
        return cls.format_RRY("Raises", ps)

    @classmethod
    def format_Yields(cls, ps, compact):
        return cls.format_RRY("Yields", ps)

    @classmethod
    def format_Returns(cls, ps, compact):
        return cls.format_RRY("Returns", ps)

    @classmethod
    def format_Attributes(cls, ps, compact):
        return cls.format_RRY("Attributes", ps)

    @classmethod
    def format_RRY(cls, name, ps):
        out = name + "\n"
        out += "-" * len(name) + "\n"

        if name == "Returns":
            if len(ps) > 1:
                # do heuristic to check we actually have a description list and not a paragraph
                pass

        for i, p in enumerate(ps):
            # if i:
            #    out += "\n"
            if p.name and p.type:
                out += f"""{p.name.strip()} : {p.type.strip()}\n"""
            elif p.name:
                out += f"""{p.name.strip()}\n"""
            else:
                out += f"""{p.type.strip()}\n"""
            if p.desc:
                out += indent("\n".join(p.desc), "    ")
                out += "\n"

        return out


def dedend_docstring(docstring):
    import textwrap

    lines = docstring.splitlines()
    if len(lines) >= 2:
        l0, *lns = docstring.split("\n")
        l0 = textwrap.dedent(l0)
        lns = textwrap.dedent("\n".join(lns)).split("\n")
        docstring = [l0] + lns
    else:
        docstring = textwrap.dedent(docstring).split("\n")
    return "\n".join(docstring)


def compute_new_doc(docstr, fname, *, level, compact, meta, func_name):
    """
    compute a new docstring that shoudl be numpydoc compliant.

    Parameters
    ----------
    docstr : str
        docstring to reformat
    fname : str
        filename of the file beign reformatted (for error messages)
    level : int
        indentation level
    compact : bool
        use compact formating in definition list.
    meta : list
        meta info about the function to verify docstrings, for example
        list of parameters.
    func_name : str
        function name for debug.

    Returns
    -------
    str
        new docstring
    Numpydoc
        parsed numpydoc object

    """
    INDENT = level * " "
    NINDENT = "\n" + INDENT
    original_docstr = docstr
    if len(docstr.splitlines()) <= 1:
        return "", NumpyDocString("")
    shortdoc = bool(docstr.splitlines()[0].strip())
    short_with_space = False
    if not docstr.startswith(NINDENT):
        if original_docstr[0] == " ":
            short_with_space = True

    long_end = True
    long_with_space = True
    if original_docstr.splitlines()[-1].strip():
        long_end = False
    if original_docstr.splitlines()[-2].strip():
        long_with_space = False

    doc = NumpyDocString(dedend_docstring(docstr))
    doc.normalize()
    if (params := doc["Parameters"]) and meta:

        a = [o.strip() for p in params for o in p.name.split(",")]
        if meta[0] in ["self", "cls"]:
            meta = meta[1:]
        doc_missing = set(meta) - set(a)
        doc_extra = {x for x in set(a) - set(meta) if not x.startswith("*")}
        if len(doc_missing) == len(doc_extra) == 1:

            print(fname)
            for i, p in enumerate(doc["Parameters"]):
                if p.name in doc_extra:
                    name = p.name
                    new_name = list(doc_missing)[0]
                    doc["Parameters"][i] = nds.Parameter(new_name, *p[1:])
                    print("renamed", name, "to", list(doc_missing)[0])
                    break
            else:
                print("  could not fix:", doc_missing, doc_extra)
        else:
            if doc_missing and doc_extra:
                print(fname)
                print("  missing:", doc_missing)
                print("  extra:", doc_extra)
            elif doc_missing or doc_extra:
                print(fname, func_name)
                print("  missing:", doc_missing)
                print("  extra:", doc_extra)

    fmt = ""
    start = True
    # ordered_section is a local patch to that records the docstring order.
    if compact:
        df = SectionFormatter(conf={"F100": "B"})
    else:
        df = SectionFormatter(conf={"F100": "A"})
    for s in getattr(doc, "ordered_sections", doc.sections):
        if doc[s]:
            f = getattr(df, "format_" + s.replace(" ", "_"))
            res = f(doc[s], compact)
            if not res:
                continue
            if not start:
                fmt += "\n"
            start = False
            fmt += res
    fmt = indent(fmt, INDENT) + INDENT

    # hack to detect if we have seen a header section.
    if "----" in fmt or True:
        if long_with_space:
            fmt = fmt.rstrip(" ") + NINDENT
        else:
            fmt = fmt.rstrip(" ") + INDENT
    if shortdoc:
        fmt = fmt.lstrip()
        if short_with_space:
            fmt = " " + fmt
    else:
        fmt = "\n" + fmt
    if not long_end:
        fmt = fmt.rstrip()
    assert fmt
    # we can't just do that as See Also and a few other would be sphinxified.
    # return indent(str(doc),'    ')+'\n    '
    return fmt, doc


def main():
    import argparse

    parser = argparse.ArgumentParser(description="reformat the docstrigns of some file")
    parser.add_argument(
        "paths",
        metavar="path",
        type=str,
        nargs="+",
        help="Files or folder to reformat",
    )
    parser.add_argument(
        "--context",
        metavar="context",
        type=int,
        default=3,
        help="Number of context lines in the diff",
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="Lift some safety feature (don't fail if updating the docstring is not indempotent",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the list of files/lines number and exit with a non-0 exit status, Use it for CI.",
    )
    parser.add_argument(
        "--no-diff",
        action="store_false",
        dest="print_diff",
        help="Do not print the diff",
    )
    parser.add_argument(
        "--no-black",
        action="store_false",
        dest="run_black",
        help="Do not run black on examples",
    )
    parser.add_argument("--no-color", action="store_false", dest="do_highlight")
    parser.add_argument("--compact", action="store_true", help="Please ignore")
    parser.add_argument(
        "--write",
        dest="write",
        action="store_true",
        help="Try to write the updated docstring to the files",
    )

    args = parser.parse_args()
    global BLACK_REFORMAT
    if args.run_black:
        BLACK_REFORMAT = True
    else:
        BLACK_REFORMAT = False
    to_format = []
    from pathlib import Path

    for f in args.paths:
        p = Path(f)
        if p.is_dir():
            for sf in p.glob("**/*.py"):
                to_format.append(sf)
        else:
            to_format.append(p)

    need_changes = []
    for file in to_format:
        # print(file)
        try:
            with open(file, "r") as f:
                data = f.read()
        except Exception as e:
            # continue
            continue
            raise RuntimeError(f"Fail reading {file}") from e

        tree = ast.parse(data)
        new = data

        # funcs = [t for t in tree.body if isinstance(t, ast.FunctionDef)]
        funcs = NodeVisitor()
        funcs.visit(tree)
        funcs = funcs.items
        for i, (func, meta) in enumerate(funcs[:]):
            # print(i, "==", func.name, "==")
            try:
                e0 = func.body[0]
                if not isinstance(e0, ast.Expr):
                    continue
                # e0.value is _likely_ a Constant node.
                docstring = e0.value.s
                func_name = func.name
            except AttributeError:
                continue
            if not isinstance(docstring, str):
                continue
            start, nindent, stop = (
                func.body[0].lineno,
                func.body[0].col_offset,
                func.body[0].end_lineno,
            )
            # if not docstring in data:
            #    print(f"skip {file}: {func.name}, can't do replacement yet")
            try:
                new_doc, d_ = compute_new_doc(
                    docstring,
                    file,
                    level=nindent,
                    compact=args.compact,
                    meta=meta,
                    func_name=func_name,
                )
                if not args.unsafe:
                    ff = new_doc

                    d2 = NumpyDocString(dedend_docstring(new_doc))
                    if not d2._parsed_data == d_._parsed_data:
                        secs1 = {
                            k: v
                            for k, v in d2._parsed_data.items()
                            if v != d_._parsed_data[k]
                        }
                        secs2 = {
                            k: v
                            for k, v in d_._parsed_data.items()
                            if v != d2._parsed_data[k]
                        }
                        raise ValueError(
                            "Numpydoc parsing seem to differ after reformatting, this may be a reformatting bug. Rerun with `velin --unsafe "
                            + str(fname)
                            + "`\n"
                            + str(secs1)
                            + "\n"
                            + str(secs2),
                        )
            except Exception as e:
                print(f"somethign went wrong with {file}:{docstring}")
                raise
                continue
            if not docstring.strip():
                print("DOCSTRING IS EMPTY !!!", func.name)
            # test(docstring, file)
            if new_doc.strip() and new_doc != docstring:
                need_changes.append(str(file) + f":{start}:{func.name}")
                if ('"""' in new_doc) or ("'''" in new_doc):
                    print(
                        "SKIPPING", file, func.name, "triple quote not handled", new_doc
                    )
                else:
                    # if docstring not in new:
                    #    print("ESCAPE issue:", docstring)
                    new = new.replace(docstring, new_doc)

            # test(docstring, file)
        if new != data:

            dold = data.splitlines()
            dnew = new.splitlines()
            diffs = list(
                difflib.unified_diff(
                    dold, dnew, n=args.context, fromfile=str(file), tofile=str(file)
                ),
            )

            if args.print_diff and not args.write:
                code = "\n".join(diffs)

                if args.do_highlight:
                    from pygments import highlight
                    from pygments.lexers import DiffLexer
                    from pygments.formatters import TerminalFormatter

                    code = highlight(code, DiffLexer(), TerminalFormatter())

                print(code)
            if args.write:
                with open(file, "w") as f:
                    f.write(new)
    if args.check:
        if need_changes:
            sys.exit(
                "Some files/functions need updates:\n - " + "\n - ".join(need_changes)
            )
        else:
            sys.exit(0)
