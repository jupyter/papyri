#!/usr/bin/env python
#
# Urwid tour.  It slices, it dices..
#    Copyright (C) 2004-2011  Ian Ward
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Urwid web site: http://excess.org/urwid/

"""
Urwid tour.  Shows many of the standard widget types and features.
"""
import sys
from typing import List

import urwid
import urwid.raw_display
import urwid.web_display

# from there import syslogprint as LOG
from urwid import Text
from urwid.canvas import CompositeCanvas, apply_text_layout
from urwid.command_map import CURSOR_DOWN, CURSOR_LEFT, CURSOR_RIGHT, CURSOR_UP
from urwid.text_layout import calc_coords
from urwid.widget import LEFT, SPACE

from papyri.crosslink import RefInfo, encoder
from papyri.config import ingest_dir
from papyri.myst_ast import MParagraph


class Link:
    def __init__(self, attr, text, cb):
        self.attr = attr
        self.text = text
        self.cb = cb

    def selectable(self):
        return True


class TextWithLink(urwid.Text):
    # _selectable = True
    # ignore_focus = False
    signals = ["change", "postchange"]

    def get_cursor_coords(self, size):
        """
        Return the (*x*, *y*) coordinates of cursor within widget.

        >>> Edit("? ","yes").get_cursor_coords((10,))
        (5, 0)
        """

        if not self._focusable:
            return None

        (maxcol,) = size

        # LOG(len(self.get_text()[0]))
        trans = self.get_line_translation(maxcol)

        current_len = 0
        k = 0
        for item in self.markup:
            if isinstance(item, Link):
                if k == self.link_index:
                    break
                else:
                    k += 1
                current_len += len(item.text)
            else:
                if isinstance(item, tuple):
                    assert len(item) == 2
                    item = item[1]
                    if isinstance(item, list):
                        for it in item:
                            assert isinstance(it, str)
                        current_len += sum(len(x) for x in item)
                    elif isinstance(item, str):
                        current_len += len(item)
                    else:
                        assert False, (repr(item), repr(self.markup))

                elif isinstance(item, str):
                    current_len += len(item)

        # LOG("FOCUS at pos", current_len)
        x, y = calc_coords(self.get_text()[0], trans, current_len)

        return (x, y)

    def compute_focused(self, markup, focus):
        nm = []
        k = 0
        for item in markup:
            if isinstance(item, Link):
                if k == self.link_index and focus:
                    nm.append(("link_selected", item.text))
                else:
                    nm.append((item.attr, item.text))
                k += 1
            else:
                nm.append(item)

        self.max_links = k
        return nm

    def __init__(self, markup, align=LEFT, wrap=SPACE, layout=None, cb=None):
        self.link_index = 0
        self.max_links = None

        self.markup = markup

        self._focusable = len([x for x in markup if isinstance(x, Link)]) > 0
        self._selectable = self._focusable
        self.ignore_focus = not self._focusable

        self.__super.__init__(
            self.compute_focused(markup, False), align=LEFT, wrap=SPACE, layout=None
        )

    def keypress(self, size, key):
        if not self._focusable:
            return key
        text, attr = self.get_text()
        if self._command_map[key] in (CURSOR_LEFT, CURSOR_UP):
            self.link_index -= 1
            if self.link_index < 0:
                self.link_index = 0
                self.set_text(self.compute_focused(self.markup, True))
                # self._invalidate()
                return "up"
            self._invalidate()
            return None
        elif self._command_map[key] in (CURSOR_RIGHT, CURSOR_DOWN):
            self.link_index += 1
            self.set_text(self.compute_focused(self.markup, True))
            if self.link_index >= self.max_links:
                self.link_index = self.max_links - 1
                self._invalidate()
                return "down"
            self._invalidate()
            return None
        elif key == "enter":
            k = 0
            for it in self.markup:
                if isinstance(it, Link):
                    if k == self.link_index:
                        it.cb()
                        return None
                    k += 1
            assert False

        else:
            self._invalidate()
            return key

    def _render(self, size, focus=False):
        """
        Render contents with wrapping and alignment.  Return canvas.

        See :meth:`Widget.render` for parameter details.

        >>> Text(u"important things").render((18,)).text # ... = b in Python 3
        [...'important things  ']
        >>> Text(u"important things").render((11,)).text
        [...'important  ', ...'things     ']
        """
        (maxcol,) = size
        self.set_text(self.compute_focused(self.markup, focus))
        text, attr = self.get_text()
        # assert isinstance(text, unicode)
        trans = self.get_line_translation(maxcol, (text, attr))

        return apply_text_layout(text, attr, trans, maxcol)

    def render(self, size, focus=False):
        """
        Render edit widget and return canvas.  Include cursor when in
        focus.

        >>> c = Edit("? ","yes").render((10,), focus=True)
        >>> c.text # ... = b in Python 3
        [...'? yes     ']
        >>> c.cursor
        (5, 0)
        """
        (maxcol,) = size
        self._shift_view_to_cursor = bool(focus)

        canv = self._render(size, focus)
        if focus:
            canv = CompositeCanvas(canv)
            canv.cursor = self.get_cursor_coords(size)

        # .. will need to FIXME if I want highlight to work again
        # if self.highlight:
        #    hstart, hstop = self.highlight_coords()
        #    d.coords['highlight'] = [ hstart, hstop ]
        return canv


blank = urwid.Divider()


def dedup(l):
    acc = []
    bk = False
    for item in l:
        if item is blank:
            if bk is not True:
                acc.append(item)
            else:
                acc.append(Text("<...>"))

            bk = True
        else:
            bk = False
            acc.append(item)
    return acc


def load(file_path, walk, qa, gen_content, frame):
    blob = encoder.decode(file_path.read_bytes())
    assert hasattr(blob, "arbitrary")
    for i in gen_content(blob, frame):
        walk.append(i)


def guess_load(rough, walk, gen_content, stack, frame):
    stack.append(rough)

    candidates = list(ingest_dir.glob(f"*/*/module/{rough}"))
    if candidates:
        for _q in range(len(walk)):
            walk.pop()
        try:
            load(candidates[0], walk, rough, gen_content, frame)
            return True
        except Exception as e:
            raise ValueError(str(candidates)) from e
    return False


class Renderer:
    def __init__(self, frame, walk, gen_content, stack):
        self.frame = frame
        self.walk = walk
        self.gen_content = gen_content
        self.stack = stack

    def cb(self, value):
        # self.frame.footer = urwid.AttrWrap(
        #    urwid.Text(["Enter ?...: ", str(value)]), "header"
        # )
        if value.__class__.__name__ == "RefInfo":
            guess_load(value.path, self.walk, self.gen_content, self.stack, self.frame)
        elif isinstance(value, str):
            guess_load(value, self.walk, self.gen_content, self.stack, self.frame)

    def render(self, obj):
        name = obj.__class__.__name__
        method = getattr(self, "render_" + name, None)
        if not method:
            return urwid.Text(("unknown", "<" + obj.__class__.__name__ + ">"))

        return method(obj)

    def render_Math(self, d):
        cont = "".join(d.value)
        from flatlatex import converter

        c = converter()
        return ("math", c.convert(cont))

    def render_Directive(self, d):
        cont = "".join(d.value)
        if d.role == "math":
            assert False
            from flatlatex import converter

            c = converter()

            return ("math", c.convert(cont))
        return Text(("directive", f"{d.domain}:{d.role}:`{cont}`"))

    def render_Example(self, ex):
        acc = []
        for line in ex.lines:
            acc.append(Text(line._line))
        a = urwid.Pile(acc)
        acc = []
        for line in ex.ind:
            acc.append(Text(line._line))
        b = urwid.Padding(urwid.Pile(acc), left=4)
        return urwid.Pile([a, b])

    def render_Link(self, link):
        if link.reference.kind == "local":
            return Text(("local", link.value))
        return TextWithLink("link", link.value, lambda: self.cb(link.reference))

    def render_BlockQuote(self, quote):
        return urwid.Padding(
            urwid.Pile([self.render(c) for c in quote.children]), left=4, right=4
        )

    def render_MAdmonition(self, adm):
        kind = adm.kind
        if hasattr(adm, "title"):
            title = (f"{kind} : {adm.title}",)
        else:
            title = (f"{kind.capitalize()}")
        if kind == "versionchanged":
            title = "Changed in Version " + adm.title
        if kind == "versionadded":
            title = "Added in Version " + adm.title
        if kind == "deprecated":
            title = "Deprecated since " + adm.title
        return urwid.Padding(
            urwid.LineBox(
                urwid.Pile([self.render(c) for c in adm.children]),
                title=title,
                title_align="left",
            ),
        )

    def render_MText(self, text):
        return urwid.Text(text.value)

    def render_MParagraph(self, paragraph):
        stuff = [self.render(c) for c in paragraph.children]
        return urwid.Pile(stuff)

    def render_BlockMath(self, math):
        from flatlatex import converter

        c = converter()
        return urwid.Padding(urwid.Text(("math", c.convert(math.value))), left=2)

    def render_BlockDirective(self, directive):
        raise NotImplementedError("We should nt have block directive in the end")

    def render_SeeAlsoItem(self, sa):
        return urwid.Pile(
            [
                TextWithLink(
                    [
                        Link(
                            "link" if sa.name.exists else "link-broken",
                            sa.name.value,
                            lambda: self.cb(sa.name.reference),
                        )
                    ]
                ),
                urwid.Padding(
                    urwid.Pile([self.render(x) for x in sa.descriptions]), left=2
                ),
            ]
        )

    def render_Verbatim(self, verb):
        return ("verbatim", verb.value)

    def render_BlockVerbatim(self, verb):
        acc = [Text(("verbatim", verb.value))]
        return urwid.Padding(urwid.Pile(acc), left=4)

    def render_Paragraph(self, paragraph):
        if any([isinstance(x, MParagraph) for x in paragraph.children]):
            assert len(paragraph.children) == 1
            return self.render_Paragraph(paragraph.children[0])

        cc = paragraph.children
        if not cc:
            return urwid.Text("EMPTY")
        rr = None
        try:
            rr = [TextWithLink([self.render(o) for o in paragraph.children])]
            return urwid.Pile(rr)
        except Exception:
            raise ValueError(cc, rr)

    def render_Section(self, section):
        if section.title:
            acc = [Text(("section", section.title))]
        else:
            acc = []
        for c in section.children:
            acc.append(self.render(c))
            # acc.append(Text("<Section Blank>"))
            acc.append(blank)

        return urwid.Padding(
            urwid.Pile(dedup(acc)),
            left=2,
            right=2,
        )

    def render_DefList(self, deflist):
        p = [blank]
        for c in deflist.children:
            assert c.__class__.__name__ == "DefListItem", c.__class__.__name__
            res = self.render(c)
            assert isinstance(res, list)
            p.extend(res)
        return urwid.Pile(p)

    def render_EnumeratedList(self, elist):
        p = [blank]
        for i, c in enumerate(elist.children, start=1):
            res = self.render(c)
            p.extend([urwid.Columns([(3, urwid.Text(f"{i}.")), res])])
        return urwid.Pile(p)

    def render_DefListItem(self, item):
        return [
            self.render(item.dt),
            # urwid.Button(str(item.dt)),
            urwid.Padding(
                urwid.Pile([self.render(p) for p in item.dd]),
                left=2,
            ),
            blank,
        ]

    def render_Fig(self, fig):
        def show_fig(name):
            cand = next(ingest_dir.glob(f"*/*/assets/{name}"))
            import subprocess

            subprocess.Popen(
                ["qlmanage", "-p", cand],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if sys.platform == "darwin":

            def _cb():
                show_fig(fig.value)

            msg = "Open with quicklook"
        else:

            def _cb():
                pass

            msg = "Open in separate window (Not implemented on this platform)"

        return TextWithLink(
            [
                ("", "Figure not available in terminal : "),
                Link(
                    "verbatim",
                    msg,
                    _cb,
                ),
            ]
        )

    def render_Code2(self, code):
        # entries/out/ce_status

        def insert_prompt(entries):
            yield (
                "verbatim",
                ">>>",
                # lambda: self.cb("likely copy content to clipboard"),
            )
            yield (None, " ")
            for e in entries:
                type_ = e.type
                maybe_link = e.link
                if maybe_link.__class__.__name__ == "Link":
                    assert isinstance(maybe_link.reference, RefInfo)
                    yield Link(
                        "pyg-" + str(type_),
                        maybe_link.value,
                        (lambda m: (lambda: self.cb(m.reference)))(maybe_link),
                    )
                else:
                    if maybe_link == "\n":
                        yield (None, "\n")
                        yield ("verbatim", "... ")
                    else:
                        yield ("pyg-" + str(type_), f"{maybe_link}")

        return urwid.Padding(
            urwid.Pile(
                [TextWithLink([x for x in insert_prompt(code.entries)])]
                + ([Text(code.out)] if code.out else []),
            ),
            left=2,
        )

    def render_Code(self, code):
        # entries/out/ce_status

        def insert_prompt(entries):
            yield Link(
                "verbatim",
                ">>>",
                lambda: self.cb("likely copy content to clipboard"),
            )
            yield (None, " ")
            for txt, _ref, css in entries:
                if txt == "\n":
                    yield (None, "\n")
                    yield ("verbatim", "... ")
                else:
                    yield ("pyg-" + str(css), f"{txt}")

        return urwid.Padding(
            urwid.Pile(
                [TextWithLink([x for x in insert_prompt(code.entries)])]
                + ([Text(code.out)] if code.out else []),
            ),
            left=2,
        )

    def render_Parameters(self, parameters):
        return urwid.Pile([self.render_Param(c) for c in parameters.children])

    def render_Param(self, param):
        return urwid.Pile(
            [
                Text(
                    [
                        ("param", param.param),
                        " : ",
                        ("type", param.type_),
                    ]
                ),
                urwid.Padding(
                    urwid.Pile([self.render(d) for d in param.desc]),
                    left=3,
                    right=2,
                    min_width=20,
                ),
            ]
        )

    def render_SignatureNode(self, sig):
        if "Empty" in str(sig.return_annotation):
            annotation = "None"
        else:
            annotation = ("signature", sig.return_annotation)
        return [
            ("signature", "("),
            [("param", f"{p.name}, ") for p in sig.parameters],
            ("signature", ")"),
            ("signature", " -> "),
            annotation,
        ]


def main(qualname: str):
    if not isinstance(qualname, str):
        from types import ModuleType

        if isinstance(qualname, ModuleType):
            qualname = qualname.__name__
        else:
            qualname = qualname.__module__ + "." + qualname.__qualname__

    # import json
    # data = json.loads(file_path.read_text())
    # data

    def gen_content(blob, frame):
        R = Renderer(frame, walk, gen_content, stack)
        doc = []
        doc.append(blank)
        if blob.signature:
            doc.append(Text(("section", "Signature")))
            doc.append(blank)
            doc.append(
                Text(
                    [
                        ("signature", blob.signature.kind),
                        (" "),
                        ("bb", qualname),
                        (" "),
                        R.render(blob.signature),
                    ]
                ),
            )
            doc.append(blank)

        for k, v in blob.content.items():
            if not v.empty():
                if k not in ["Summary", "Extended Summary"]:
                    doc.append(Text([("section", k)]))
                # doc.append(Text("<Blank InnerSec>"))
                doc.append(blank)
                doc.append(R.render(v))
        if blob.see_also:
            doc.append(Text(("section", "See Also")))
            doc.append(blank)
            for s in blob.see_also:
                doc.append(urwid.Padding(R.render(s), left=2))
                doc.append(blank)

        if not blob.example_section_data.empty():
            doc.append(Text(("section", "Examples")))
            doc.append(blank)
            doc.append(R.render(blob.example_section_data))

        if blob.item_type and ("module" in blob.item_type):
            for s in blob.arbitrary:
                doc.append(R.render(s))

        if []:  # todo backrefs
            doc.append(Text(("section", "Back References")))
            doc.append(Text("All the following items Refer to this page:"))
            for b in []:  # backrefs:
                doc.append(
                    urwid.Padding(
                        TextWithLink(
                            [Link("link", b.path, (lambda x: lambda: R.cb(x))(b))]
                        ),
                        # TextWithLink([Link("param", b, lambda: R.cb(b))]),
                        left=2,
                    )
                )
        doc = dedup(doc)

        doc.append(blank)
        doc.append(blank)
        doc.append(blank)

        def cb(value):
            def callback():
                assert isinstance(value, str)
                frame.footer = urwid.AttrWrap(
                    urwid.Text(["Enter ?...: ", value]), "header"
                )

            return callback

        return doc

    stack: List[str] = []

    walk = urwid.SimpleListWalker([])
    listbox = urwid.ListBox(walk)
    frame = urwid.Frame(urwid.AttrWrap(listbox, "body"))  # , header=header)
    frame.footer = urwid.AttrWrap(
        urwid.Text(
            "q: quit | ?: classic IPython help screen | Arrow/Click: focus links & navigate | enter: follow link"
        ),
        "header",
    )

    found = guess_load(qualname, walk, gen_content, stack, frame)
    if not found:
        return False

    # header = urwid.AttrWrap(Text("numpy.geomspace"), "header")
    #    'black'
    #    'dark red'
    #    'dark green'
    #    'brown'
    #    'dark blue'
    #    'dark magenta'
    #    'dark cyan'
    #    'light gray'
    #    'dark gray'
    #    'light red'
    #    'light green'
    #    'yellow'
    #    'light blue'
    #    'light magenta'
    #    'light cyan'
    #    'white'

    palette = [
        ("body", "default", "default", "standout"),
        ("reverse", "light gray", "black"),
        ("header", "white", "dark blue", "bold"),
        ("bb", "bold", "default", ("standout", "underline")),
        ("important", "dark red,bold", "default", ("standout", "underline")),
        ("editfc", "white", "dark blue", "bold"),
        ("editbx", "light gray", "dark blue"),
        ("editcp", "black", "light gray", "standout"),
        ("bright", "dark gray", "light gray", ("bold", "standout")),
        ("buttn", "black", "dark cyan"),
        ("buttnf", "white", "dark blue", "bold"),
        ("verbatim", "brown", "", "bold"),
        ("emph", "dark blue", "", "underline"),
        ("strgon", "dark blue", "", "bold"),
        # ("link", "dark red,bold", "default", ("standout", "underline")),
        ("local", "light magenta", "", "bold"),
        ("link", "dark green,underline", "", "bold"),
        ("link_selected", "dark green,bold", "", "bold"),
        ("link_selected", "black,bold", "white"),
        ("link-broken", "dark red,strikethrough", "", "bold"),
        ("type", "dark cyan", "", "bold"),
        ("signature", "yellow,bold", "", "bold"),
        ("param", "dark blue", "", "bold"),
        ("section", "dark magenta,bold", "", "bold"),
        ("unknown", "white", "dark red", "bold"),
        ("directive", "white", "dark red", "bold"),
        ("math", "dark magenta,italics", "", "bold"),
        # pygments
        ("pyg-o", "dark blue", "", "bold"),  # operator (+, .)
        ("pyg-mi", "dark red", "", "bold"),  # number literal 12, 55
        ("pyg-kc", "dark green", "", "bold"),
        ("pyg-nb", "white", "", "bold"),
        ("pyg-kn", "dark green", "", "bold"),  # keyword import
        ("pyg-nn", "dark blue", "", "bold"),  # name
        ("pyg-k", "dark green", "", "bold"),  # keyword as
        ("pyg-s2", "dark green", "", "bold"),  # strings, like "this is a string s2"
        ("pyg-sa", "dark green", "", "bold"),  # string brefixes like b"", u"" r""
    ]

    # use appropriate Screen class
    if urwid.web_display.is_web_request():
        screen = urwid.web_display.Screen()
    else:
        screen = urwid.raw_display.Screen()
    found = True

    def unhandled(key):
        nonlocal found
        if key == "?":
            found = False
            raise urwid.ExitMainLoop()
        elif key == "q":
            raise urwid.ExitMainLoop()
        elif key == "backspace":
            if len(stack) >= 2:
                stack.pop()
                old = stack.pop()
                guess_load(old, walk, gen_content, stack, frame)

    urwid.MainLoop(frame, palette, screen, unhandled_input=unhandled).run()
    return found


def setup() -> None:
    urwid.web_display.set_preferences("Urwid Tour")
    # try to handle short web requests quickly
    if urwid.web_display.handle_short_request():
        return
    import sys

    target: str
    target = sys.argv[1]
    assert isinstance(target, str)
    res = main(target)
    print(res)


if "__main__" == __name__ or urwid.web_display.is_web_request():
    setup()
