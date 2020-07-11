import ast
import sys
import difflib
from textwrap import indent

import numpy as np

import numpydoc.docscrape


def w(orig):
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


class DocstringFormatter:
    @classmethod
    def format_Summary(self, s):
        if len(s) == 1 and not s[0].strip():
            return ""
        return "\n".join(s) + "\n"

    @classmethod
    def format_Extended_Summary(self, es):
        return "\n".join(es) + "\n"

    @classmethod
    def _format_ps(cls, name, ps):

        out = name + "\n"
        out += "-" * len(name) + "\n"
        for p in ps:
            if p.type:
                out += f"""{p.name} : {p.type}\n"""
            else:
                out += f"""{p.name}\n"""
            if p.desc:
                out += indent("\n".join(p.desc), "    ")
                out += "\n"
        return out

    @classmethod
    def format_Parameters(cls, ps):
        return cls._format_ps("Parameters", ps)

    @classmethod
    def format_Other_Parameters(cls, ps):
        return cls._format_ps("Other Parameters", ps)

    @classmethod
    def format_See_Also(cls, sas):
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
            for ref,type_ in a:
                if not _first:
                    out += ', '
                if type_ is not None:
                    out += f":{type_}:`{ref}`"
                else:
                    out += f"{ref}"
                _first = False

            if desc:
                if len(a) > 1:
                    out += f"\n    {desc}"
                else:
                    out += f" : {desc}"
            for rd in rest_desc:
                out += "\n    " + rd
            out += "\n"
        return out

    @classmethod
    def format_References(cls, lines):
        out = "References\n"
        out += "----------\n"
        out += "\n".join(lines)
        out += "\n"
        return out

    @classmethod
    def format_Notes(cls, lines):
        out = "Notes\n"
        out += "-----\n"
        out += "\n".join(lines)
        out += "\n"
        return out

    @classmethod
    def format_Examples(cls, lines):
        out = "Examples\n"
        out += "--------\n"
        out += "\n".join(lines)
        out += "\n"
        return out

    @classmethod
    def format_Raises(cls, ps):
        return cls.format_RRY("Raises", ps)

    @classmethod
    def format_Yields(cls, ps):
        return cls.format_RRY("Yields", ps)

    @classmethod
    def format_Returns(cls, ps):
        return cls.format_RRY("Returns", ps)

    @classmethod
    def format_RRY(cls, name, ps):
        out = name + "\n"
        out += "-" * len(name) + "\n"

        for p in ps:
            if p.name:
                out += f"""{p.name} : {p.type}\n"""
            else:
                out += f"""{p.type}\n"""
            if p.desc:
                out += indent("\n".join(p.desc), "    ")
                out += "\n"
        return out


def test(docstr, fname):
    fmt = compute_new_doc(docstr, fname)
    dold = docstr.splitlines()
    dnew = fmt.splitlines()
    diffs = list(difflib.unified_diff(dold, dnew, n=100, fromfile=fname, tofile=fname),)
    if diffs:
        from pygments import highlight
        from pygments.lexers import DiffLexer
        from pygments.formatters import TerminalFormatter

        code = "\n".join(w(diffs))
        hldiff = highlight(code, DiffLexer(), TerminalFormatter())

        print(indent(hldiff, " |   ", predicate=lambda x: True))


def compute_new_doc(docstr, fname):
    original_docstr = docstr
    if len(docstr.splitlines()) == 1:
        return ""
    shortdoc=False
    if not docstr.startswith("\n    "):
        docstr = "\n    " + docstr
        shortdoc = True
        print(shortdoc, docstr)
    doc = numpydoc.docscrape.NumpyDocString(docstr)

    fmt = ""
    start = True
    for s in doc.sections:
        if doc[s]:
            f = getattr(
                DocstringFormatter, "format_" + s.replace(" ", "_"), lambda x: s
            )
            if not start:
                fmt += "\n"
            start = False
            fmt += f(doc[s])

    fmt = indent(fmt, "    ") + "    "
    if shortdoc:
        fmt = fmt.lstrip()
    else:
        fmt = '\n'+fmt
    return fmt


def main():
    import argparse

    parser = argparse.ArgumentParser(description="reformat the docstrigns of some file")
    parser.add_argument("files", metavar="files", type=str, nargs="+", help="TODO")
    parser.add_argument("--context", metavar="context", type=int, default=3)
    parser.add_argument(
        "--write", dest="write", action="store_true", help="print the diff"
    )

    args = parser.parse_args()
    for file in args.files:
        #print(file)
        with open(file, "r") as f:
            data = f.read()

        tree = ast.parse(data)
        new = data

        funcs = [t for t in tree.body if isinstance(t, ast.FunctionDef)]
        for i, func in enumerate(funcs[:]):
            # print(i, "==", func.name, "==")
            try:
                docstring = func.body[0].value.s
            except AttributeError:
                continue
            if not isinstance(docstring, str):
                continue
            start, nindent, stop = (
                func.body[0].lineno,
                func.body[0].col_offset,
                func.body[0].end_lineno,
            )
            if not docstring in data:
                print(f"skip {file}: {func.name}, can't do replacement yet")

            new_doc = compute_new_doc(docstring, file)
            # test(docstring, file)
            if new_doc:
                if ('"""' in new_doc) or ("'''" in new_doc):
                    print(
                        "SKIPPING", file, func.name, "triple quote not handled", new_doc
                    )
                else:
                    new = new.replace(docstring, new_doc)

            # test(docstring, file)
        if new != data:
            dold = data.splitlines()
            dnew = new.splitlines()
            diffs = list(
                difflib.unified_diff(dold, dnew, n=args.context, fromfile=file, tofile=file),
            )
            from pygments import highlight
            from pygments.lexers import DiffLexer
            from pygments.formatters import TerminalFormatter

            if not args.write:
                code = "\n".join(diffs)
                hldiff = highlight(code, DiffLexer(), TerminalFormatter())

                print(hldiff)
            else:
                with open(file, "w") as f:
                    f.write(new)
