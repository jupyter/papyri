"""
French for Vellum

> Vellum is prepared animal skin or "membrane", typically used as a material for
> writing on. Parchment is another term > for this material, and if vellum is
> distinguished from this, it is by vellum being made from calfskin, as opposed to
> that from other animals,[1] or otherwise being of higher quality

Tools to automatically reformat docstrings based using numpydoc format.

"""

import sys
from pathlib import Path
import textwrap

from numpydoc.docscrape import NumpyDocString

__version__ = "0.0.2"


def parse(input):
    """
    parse an input string into token/tree.

    For now only return a list of tokens

    """
    tokens = []
    for l in input.splitlines():
        tokens.extend(l.split(" "))
    return tokens


def transform(tokens):
    """
    Accumulate tokens in lines.

    Add token (and white spaces) to a line until it overflow 80 chars.
    """
    lines = []
    current_line = []
    for t in tokens:
        if sum([len(x) + 1 for x in current_line]) + len(t) > 80:
            lines.append(current_line)
            current_line = []
        current_line.append(t)
    if current_line:
        lines.append(current_line)
    return lines


def format(lines):
    s = ""
    for l in lines:
        s = s + " ".join(l)
        s = s + "\n"
    return s[:-1]


def compute_indents(lines):
    """
    Given a list of lines, compute indentation in number of spaces.

    Indentation is only supported if spaces, tabs raise a NotImplementedError as we don't know how wide a tab is.
    We also treat complete blank lines as `None` indentation.


    """
    assert isinstance(lines, list)
    results = []
    for l in lines:
        s = l.lstrip()
        if not s:
            results.append(None)
            continue
        indent = len(l) - len(s)
        if "\t" in l[:indent]:
            raise NotImplementedError

        results.append(indent)
    return results


class TryNext(Exception):
    pass


class Header:
    def __init__(self, title, level):
        self.title = title
        # if str(title) not in NumpyDocString.sections.keys():
        #    print(f'??? |{title}|')
        self.level = level

    def __repr__(self):
        return f"<Header {self.level}:  {self.title} >"

    def __str__(self):
        tt = str(self.title)
        return tt + "\n" + "=" * len(tt)

    def _repr_html_(self):
        return f"<h{self.level}>" + str(self.title) + f"</h{self.level}>"

    @classmethod
    def parse(cls, lines):
        if len(lines) < 2:
            raise TryNext
        assert lines
        warnings = []
        title, _, wn = Raw.parse(lines)
        lgth = len(title.lines[0])
        if (
            len(set(lines[1])) == 1
            and len(lines[1]) != 1
            and len(lines[1]) != lgth
            and ">>>" not in lines[0]
            and "::" not in lines[1]
        ):

            warnings.append("======= WRONG LEN? ======")
            warnings.append("L0: " + lines[0])
            warnings.append("L1: " + lines[1])
            warnings.append("=========================")
        level = allunders(lines[1], lgth)
        return cls(title, level), lines[2:], wn


def allunders(line, lenght):

    if not len(set(line)) == 1 or not len(line) == lenght:
        raise TryNext
    if next(iter(set(line))) not in "-=~`":
        raise TryNext
    return 1


class Any:
    def __init__(self, line):
        assert isinstance(line, str)
        self.lines = [line]

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.lines!r} >"

    def __str__(self):
        return self.lines[0]

    @classmethod
    def parse(cls, lines):
        return cls(lines[0]), lines[1:], []


class Raw(Any):
    pass


class RawTilNextHeader:
    @classmethod
    def parse(cls, lines):
        ll = []
        for i, l in enumerate(lines):
            try:
                Header.parse(lines[i:])
                break
            except TryNext:
                pass

            ll.append(l)
        else:
            return cls(ll), [], []

        return cls(ll), lines[i:], []

    def __init__(self, items):
        assert isinstance(items, list)
        self.items = items

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.items!r} >"

    def __str__(self):
        return "\n".join(self.items)

    def _repr_html_(self):
        return """<pre>""" + str(self) + """</pre>"""


class DescriptionList:
    @classmethod
    def parse(cls, lines):
        dct = {}
        key, values = None, []
        for i, l in enumerate(lines):
            try:
                Header.parse(lines[i:])
                dct[key] = values
                break
            except TryNext:
                pass

            if not l.startswith(" "):
                dct[key] = values
                key, values = l.strip(), []
                if "," in key:
                    raise TryNext
            else:
                values.append(l)
        if None in dct:
            del dct[None]
        return cls(dct), lines[i:], []

    def __init__(self, items):
        assert isinstance(items, dict)
        self.items = items

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.items!r} >"

    def __str__(self):
        def _f(v):
            return "\n".join(v)

        return "\n".join(f"{k}:\n{_f(v)}" for k, v in self.items.items())

    def _repr_html_(self):
        def _f(v):
            return "\n".join(v)

        return (
            """<dl>"""
            + "\n".join(
                f"<dt>{k}</dt>\n<dd>{_f(v)}</dd>" for k, v in self.items.items()
            )
            + """</dl>"""
        )


class Listing:
    @classmethod
    def parse(cls, lines):
        assert "," in lines[0], lines[0]
        listing = [l.strip() for l in lines[0].split(",")]
        return cls(listing), lines[1:], []

    def __init__(self, listing):
        self.listing = listing

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.listing!r} >"

    def __str__(self):
        return ",".join([f"{k}" for k in self.listing])

    def _repr_html_(self):
        return (
            """<ul>""" + "\n".join(f"<li>{k}</li>" for k in self.listing) + """</ul>"""
        )


class Base:
    def attrs(self):
        stuff = {}
        for it in dir(self):
            if it.startswith("_"):
                continue
            att = getattr(self, it)
            if callable(att):
                continue
            stuff[it] = att
        return stuff

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            + ", ".join([k + ":" + repr(x) for k, x in self.attrs().items()])
            + ">"
        )


class EntryParser(Base):
    @classmethod
    def parse(cls, lines):
        if len(lines) == 1:
            l0, l1 = lines[0], ""
            rest = []
        elif len(lines) < 3:
            l0, l1 = lines
            rest = []
        else:
            l0, l1, *rest = lines
        indent = len(l1) - len(l1.lstrip())
        i = 0
        if indent:
            cont = [l1]
            for i, l in enumerate(rest):
                if l.startswith(" " * indent) or not l.strip():
                    cont.append(l)
                else:
                    break
            else:
                i += 1
        else:
            i = 0
            cont = []
            rest = lines[1:]
        if ":" in l0:
            try:
                head, t = [x.strip() for x in l0.split(":", maxsplit=1)]
            except ValueError:
                print("... Entry TryNext", lines[:5])
                raise TryNext
        else:
            head, t = l0.strip(), ""

        if " " in head:
            if not t and not cont:
                # print('... list of things ? ', head)
                return [cls(h.strip(), "", []) for h in head.split(",")], rest[i:]

            if "See Also" in head:
                print("-------------->", lines[:3])

        return [cls(head, t, cont)], rest[i:]

    def __init__(self, head, t, rest):
        # assert (head.strip() or t.strip() or [r.strip() for r in rest])
        self.head = head
        self.t = t
        self.rest = rest

    def __str__(self):
        r = "\n".join(self.rest)
        if r:
            extra = f"\n{r}"
        else:
            extra = ""
        return f"""{self.head}:{self.t}{extra}"""

    def _format_head(self, head, resolver):
        return resolver(head)

    def _format_type(self, key, resolver):
        if self.rest:
            return key
        else:
            return ""

    def _format_core(self, core, resolver):
        if self.rest:
            return "<pre>" + "\n".join(core) + "</pre>"
        else:
            return f"<pre>{self.t}</pre>"

    def _repr_html_(self, resolver):

        return f"<dt>{self._format_head(self.head, resolver)}:{self._format_type(self.t, resolver)}</dt><dd>{self._format_core(self.rest, resolver)}</dd>"


class DeflistParser(Base):
    def __init__(self, entries):
        self.entries = entries

    @classmethod
    def parse(cls, lines):
        ents = []
        while lines:
            if not lines[0].strip():
                lines = lines[1:]
                continue
            try:
                Header.parse(lines)
                break
            except TryNext:
                pass
            e, lines = EntryParser.parse(lines)
            ents.extend(e)

        return cls(ents), lines

    def __str__(self):
        return "\n".join(str(x) for x in self.entries)

    def _repr_html_(self, resolver=None):

        return (
            """<dl> (Deflist)"""
            + "\n".join(x._repr_html_(resolver) for x in self.entries)
            + """</dl>"""
        )


class Mapping:
    @classmethod
    def parse(cls, lines):
        mapping = {}
        k = None
        for i, l in enumerate(lines):
            if not l.strip():
                continue
            try:
                Header.parse(lines[i:])
                break
            except TryNext:
                pass
            if l.startswith(" ") and k:
                try:
                    mapping[k.strip()] += l.strip()
                except TypeError:
                    raise TryNext

            if ":" in l:
                k, v = l.split(":", maxsplit=1)
                mapping[k.strip()] = v
            elif "," not in l:
                k, v = l.strip(), None
                mapping[k] = v
            else:
                for k in l.split(","):
                    mapping[k.strip()] = None

        return cls(mapping), lines[i:], []

    def __init__(self, mapping):
        self.mapping = mapping

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.mapping!r} >"

    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.mapping.items()])

    def _format_one(self, key, resolver):
        return resolver(key)

    def _format_pair(self, key, value, resolver):
        k = self._format_one(key, resolver)
        v = value
        return f"<dt>{k}</dt>\n<dd>{v}</dd>"

    def _repr_html_(self, resolver):

        return (
            """<dl> (Mapping)"""
            + "\n".join(
                [self._format_pair(k, v, resolver) for k, v in self.mapping.items()]
            )
            + """</dl>"""
        )


class CodeBlock:
    @classmethod
    def parse(cls, lines):
        if not lines[0].startswith((">>>", "    >>>")):
            raise TryNext

        _lines = []
        for i, l in enumerate(lines):
            if not l.strip():
                break
            _lines.append(l)
        else:
            return cls(_lines), [], []
        return cls(_lines), lines[i:], []

    def __init__(self, lines):
        self.lines = lines

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.lines!r} >"

    def _repr_html_(self):
        return "<pre>" + "\n".join(self.lines) + "</pre>"

    def __str__(self):
        return "\n".join(self.lines)


class Doc:
    @classmethod
    def parse(cls, lines, *, name=None, sig=None):
        parsed = []
        cl = len(lines)
        warnings = []
        while lines:
            for t in (
                Section.parse,
                Header.parse,
                BlankLine.parse,
                CodeBlock.parse,
                Paragraph.parse,
                RawTilNextHeader.parse,
                failed,
            ):
                try:
                    node, lines_, wn = t(lines)
                    warnings.extend(wn)
                    if isinstance(node, list):
                        parsed.extend(node)
                    else:
                        parsed.append(node)
                    if len(lines_) >= len(lines):
                        raise ValueError("Could not parse", lines)
                    lines = lines_
                    break
                except TryNext:
                    pass
        return cls(parsed, name=name, sig=sig), warnings

    def __init__(self, nodes, name=None, sig=None):
        self.nodes = nodes
        self.name = name
        self.sig = sig
        self.backrefs = []

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}\n"
            + "\n".join([textwrap.indent(repr(n), "    ") for n in self.nodes])
            + "\n>"
        )

    def see_also(self):
        for i, p in enumerate(self.nodes):
            if isinstance(p, Section) and p.header.title.lines[0] == "See Also":
                break
        else:
            return []

        node = self.nodes[i + 1]
        if isinstance(node, Mapping):
            return node.mapping.keys()
        elif isinstance(node, DeflistParser):
            return [x.head for x in node.entries]
        else:
            print("not a mapping", repr(node))
            pass

    def _repr_html_(self, resolver=lambda x: None):
        base = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="https://fonts.xz.style/serve/inter.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@exampledev/new.css@1.1.2/new.min.css">
</head>
<body>
    {}
    {}
    {}
</body>
</html>
"""

        h1 = ""
        if self.name:
            h1 += self.name
        if self.sig:
            h1 += self.sig
        if h1:
            h1 = "<h1>" + h1 + "</h1>"

        def f_(it):
            return f"<a href=./{it}.html>{it}</a>"

        if self.backrefs:
            backref_repr = ",".join(self.backrefs)
            br_html = "<h1>Back references</h1>" + ", ".join(
                f_(b) for b in self.backrefs
            )
        else:
            br_html = ""

        def _resolver(k):
            if (ref := resolver(k)) is None:
                # print("could not resolve", k, f"({self.name})")
                return k
            else:
                # print("resolved", k, f"({self.name})")
                return f"<a href='{ref}.html'>{k}</a>"

        hrepr = []
        for n in self.nodes:
            if isinstance(n, (Mapping, DeflistParser)):
                hrepr.append(n._repr_html_(_resolver))
            else:
                hrepr.append(n._repr_html_())

        return base.format(h1, "\n".join(hrepr), br_html)


class Section:
    """
    start with a header, but have custom parsing because we know about it in numpydoc.
    """

    @classmethod
    def parse(cls, lines):
        warnings = []
        header, rest, wn = Header.parse(lines)
        warnings.extend(wn)
        aliases = {
            "Return": "Returns",
            "Arguments": "Parameters",
            "arguments": "Parameters",
            "additional keyword arguments:": "Parameters",
            "Other parameters": "Other Parameters",
            "Exceptions": "Raises",
        }
        ht = header.title.lines[0]
        if ht in aliases:
            warnings.append(f"Found `{ht}`, did you mean `{aliases[ht]}` ?")
            header.title.lines[0] = aliases[ht]
        ht = header.title.lines[0]
        if ht in (
            "Parameters",
            "Returns",
            "Class Attributes",
            "Options",
            "Attributes",
            "Yields",
            "Raises",
            "Exceptions",
            "Methods",
            "Warns",
            "Other Parameters",
            "Warnings",
            "Arguments",
        ):
            try:
                try:
                    core, rest, _ = Paragraph.parse(rest)
                except TryNext:
                    core, rest = DeflistParser.parse(rest)
            except TryNext:
                core, rest, wn = DescriptionList.parse(rest)
                warnings.extend(wn)
            return [cls(header), core], rest, warnings
        elif header.title.lines[0] in ("See Also", "Returns", "See also"):
            if header.title.lines[0] == "See also":
                header.title.lines[0] = "See Also"
            try:
                core, rest = DeflistParser.parse(rest)
            except TryNext:
                print("Deflist failed trying Mapping... ")
                core, rest, wn = Mapping.parse(rest)
                warnings.extend(wn)
                # core, rest, wn = DescriptionList.parse(rest)
                # warnings.extend(wn)

            return [cls(header), core], rest, warnings
        elif header.title.lines[0] in ("Examples",):
            return [cls(header)], rest, warnings
        elif header.title.lines[0] in ("Notes", "References"):
            return [cls(header)], rest, warnings

        else:
            raise ValueError(repr(header.title.lines[0]))

    def __init__(self, header):
        self.header = header

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.header!r}>"

    def __str__(self):
        return str(self.header)

    def _repr_html_(self):
        return self.header._repr_html_()


class Paragraph:
    @classmethod
    def parse(cls, lines):
        _lines = []
        l0 = lines[0]
        if not l0 or l0.startswith(" "):
            raise TryNext
        if len(lines) >= 2:
            if lines[1].startswith(" ") and lines[1].strip():
                # second line indented this _is_ a deflist
                raise TryNext
        for i, l in enumerate(lines):
            if l and not l.startswith(" "):
                _lines.append(l)
            else:
                break
        if not _lines:
            raise TryNext
        return cls(_lines), lines[i + 1 :], []

    def __init__(self, lines):
        self.lines = lines

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.lines}>"

    def __str__(self):
        return "\n".join(self.lines)

    def _repr_html_(self):
        return "<p>" + " ".join(self.lines) + "</p>"


class BlankLine:
    @classmethod
    def parse(self, lines):
        if not lines[0].strip():
            return BlankLine(), lines[1:], []
        raise TryNext

    def __init__(self):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __str__(self):
        return ""

    def _repr_html_(self):
        return ""


def failed(lines):
    raise ValueError("nothign managed to parse", lines)


def parsedoc(doc, *, name=None, sig=None):

    try:
        NumpyDocString(doc)
        lines = dedentfirst(doc).splitlines()
        return Doc.parse(lines, name=name, sig=sig)
    except Exception:
        return ""


def find_indent_blocks(lines):
    """
    Given a list of lines find _block_ by level of indentation

    - A block is considered a sequence of one or more lines, separated 
    - once one level of indentation is encounter we don't split anymore, it will be up to the sub-parser.
    - A whitespace or empty line yield another block.
    """
    if isinstance(lines, str):
        raise ValueError("split please")
    indents = compute_indents(lines)
    assert len(indents) == len(lines)
    l0 = lines[0]
    indent_level = indents[0]
    if indent_level is None:
        indent_level = 0
    assert indent_level == 0
    current_block = [l0]

    n_empty_lines = 0

    blocks = []
    for new_level, l in zip(indents[1:], lines[1:]):
        if new_level is None:
            n_empty_lines += 1
            continue

        if n_empty_lines and new_level == 0:
            current_block.extend([""] * n_empty_lines)
            n_empty_lines = 0
            blocks.append((indent_level, current_block))
            current_block = [l]
            indent_level = new_level
            continue
        if n_empty_lines:
            current_block.extend([""] * n_empty_lines)
            n_empty_lines = 0

        if indent_level == 0 and new_level:
            # start a new block
            blocks.append((0, current_block))
            current_block = [l[new_level:]]
            indent_level = new_level
            continue

        # we are already in indented blocks.
        if new_level >= indent_level:
            current_block.append(l[indent_level:])
        elif new_level < indent_level:
            blocks.append((indent_level, current_block))
            current_block = [l]
            indent_level = new_level
            continue
    if current_block:
        blocks.append((indent_level, current_block))
        current_block = []

    assert len(current_block) == 0
    return blocks


def dedentfirst(docstring):
    from textwrap import dedent

    lines = docstring.splitlines()
    ln = dedent("\n".join(lines[1:])).splitlines()
    l0 = lines[0]
    return "\n".join([dedent(l0)] + ln)


def reformat(input):
    return format(transform(parse(format(transform(parse(input))))))


# def main():
#
#    for filename in sys.argv[1:]:
#        p = Path(filename)
#
#        with p.open() as f:
#            data = f.read()
#            print(reformat)

from .ref import main


if __name__ == "__main__":
    main()
