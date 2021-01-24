#!/usr/bin/env python
# coding: utf-8
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

from papyri.crosslink import load_one

import urwid
import urwid.raw_display
import urwid.web_display

from urwid.canvas import apply_text_layout
from urwid.canvas import CompositeCanvas

from urwid.command_map import (
    command_map,
    CURSOR_LEFT,
    CURSOR_RIGHT,
    CURSOR_UP,
    CURSOR_DOWN,
    CURSOR_MAX_LEFT,
    CURSOR_MAX_RIGHT,
)


# Text alignment modes
LEFT = "left"
RIGHT = "right"
CENTER = "center"

from urwid.widget import LEFT, SPACE

from urwid import Text


class Link:
    def __init__(self, attr, text, cb):
        self.attr = attr
        self.text = text
        self.cb = cb


class TextWithLink(urwid.Text):
    _selectable = True
    ignore_focus = False
    signals = ["change", "postchange"]

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

        self.__super.__init__(
            self.compute_focused(markup, False), align=LEFT, wrap=SPACE, layout=None
        )

    def keypress(self, size, key):
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

        # .. will need to FIXME if I want highlight to work again
        # if self.highlight:
        #    hstart, hstop = self.highlight_coords()
        #    d.coords['highlight'] = [ hstart, hstop ]
        return canv


def main(ex):
    if not isinstance(ex, str):
        ex = ex.__module__ + "." + ex.__qualname__

    import os.path
    from pathlib import Path

    # import json
    # data = json.loads(file_path.read_text())
    # data

    blank = urwid.Divider()

    class Renderer:
        def __init__(self, frame):
            self.frame = frame

        def cb(self, value):
            self.frame.footer = urwid.AttrWrap(
                urwid.Text(["Enter ?...: ", str(value)]), "header"
            )
            if value.__class__.__name__ == "RefInfo":
                guess_load(value.path)
            elif isinstance(value, str):
                guess_load(value)

        def render(self, obj):
            name = obj.__class__.__name__
            method = getattr(self, "render_" + name, None)
            if not method:
                return urwid.Text(("unknown", "<" + obj.__class__.__name__ + ">"))

            return method(obj)

        def render_Directive(self, d):
            cont = "".join(d.value)
            return ("directive", f"{d.domain}:{d.role}:`{cont}`")

        def render_Words(self, words):
            return words.value

        def render_Link(self, link):
            if link.reference.kind == "local":
                return ("link", link.reference.path)
            return Link("link", link.reference.path, lambda: self.cb(link.reference))

        def render_BlockDirective(self, directive):
            inn = [
                blank,
                Text(
                    [("param", ".. " + directive.directive_name + "::")]
                    + directive.args0
                ),
            ]
            if directive.inner:
                inn.append(urwid.Padding(self.render(directive.inner), left=2)),
            return urwid.Pile(inn)

        def render_SeeAlsoItem(self, sa):
            return urwid.Pile(
                [
                    TextWithLink(
                        [Link("link", sa.name.name, lambda: self.cb(sa.name.ref))]
                    ),
                    urwid.Padding(
                        urwid.Pile([self.render(x) for x in sa.descriptions]), left=2
                    ),
                ]
            )

        def render_Verbatim(self, verb):
            return ("verbatim", verb.value)

        def render_BlockVerbatim(self, verb):
            acc = []
            for line in verb.lines:
                acc.append(Text(line._line))
            return urwid.Pile(acc)

        def render_Paragraph(self, paragraph):
            return TextWithLink([self.render(o) for o in paragraph.children])

        def render_Section(self, section):
            acc = []
            for c in section.children:
                acc.append(self.render(c))
                acc.append(blank)

            return urwid.Padding(
                urwid.Pile(acc),
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

        def render_DefListItem(self, item):
            return [
                self.render(item.dt),
                urwid.Padding(
                    self.render(item.dd),
                    left=2,
                ),
                blank,
            ]

        def render_Fig(self, code):
            return TextWithLink(
                [
                    ("", "Figure not available in terminal : "),
                    Link(
                        "verbatim",
                        "Open in os window",
                        lambda: self.cb("Not Implemented"),
                    ),
                ]
            )

        def render_Code(self, code):
            # entries/out/ce_status

            def insert_prompt(entries):
                yield Link(
                    "verbatim",
                    ">>> ",
                    lambda: self.cb("likely copy content to clipboard"),
                )
                for txt, ref, css in entries:
                    if txt == "\n":
                        yield (None, "\n")
                        yield ("verbatim", "... ")
                    else:
                        yield ("pyg-" + str(css), txt)

            return urwid.Padding(
                urwid.Pile(
                    [
                        blank,
                        TextWithLink([x for x in insert_prompt(code.entries)]),
                        Text(code.out),
                        blank,
                    ]
                ),
                left=4,
            )

        def render_Param(self, param):
            return urwid.Pile(
                [
                    TextWithLink(
                        [
                            Link("param", param.param, lambda: None),
                            # ("param", param.param),
                            " : ",
                            ("type", param.type_),
                        ]
                    ),
                    urwid.Padding(
                        urwid.Pile([self.render(d) for d in param.desc]),
                        left=2,
                        right=2,
                        min_width=20,
                    ),
                ]
            )


    def gen_content(blob, frame):
        R = Renderer(frame)
        doc = []
        doc.append(blank)
        if blob.signature:
            doc.append(Text([("signature", blob.signature)]))

        for k, v in blob.content.items():
            from papyri import take2

            if not v.empty():
                if k not in ["Summary", "Extended Summary"]:
                    doc.append(TextWithLink([Link("section", k, lambda: None)]))
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

    stack = []

    def guess_load(rough):
        stack.append(rough)
        from papyri.config import ingest_dir

        candidates = list(ingest_dir.glob(f"*/*/module/{rough}.json"))
        if candidates:
            for q in range(len(walk)):
                walk.pop()
            try:
                load(candidates[0], walk, rough)
            except Exception as e:
                raise ValueError(str(candidates)) from e

    walk = urwid.SimpleListWalker([])
    listbox = urwid.ListBox(walk)
    frame = urwid.Frame(urwid.AttrWrap(listbox, "body"))  # , header=header)

    def load(file_path, walk, qa):
        blob = load_one(file_path.read_text(), "[]", qa=qa)
        for i in gen_content(blob, frame):
            walk.append(i)

    guess_load(ex)

    # header = urwid.AttrWrap(Text("numpy.geomspace"), "header")

    palette = [
        ("body", "default", "default", "standout"),
        ("reverse", "light gray", "black"),
        ("header", "white", "dark red", "bold"),
        ("bb", "bold", "default", ("standout", "underline")),
        ("important", "dark red,bold", "default", ("standout", "underline")),
        ("link", "dark red,bold", "default", ("standout", "underline")),
        ("link_selected", "dark red,bold", "light blue"),
        ("editfc", "white", "dark blue", "bold"),
        ("editbx", "light gray", "dark blue"),
        ("editcp", "black", "light gray", "standout"),
        ("bright", "dark gray", "light gray", ("bold", "standout")),
        ("buttn", "black", "dark cyan"),
        ("buttnf", "white", "dark blue", "bold"),
        ("verbatim", "brown", "", "bold"),
        ("link", "dark green", "", "bold"),
        ("type", "dark cyan", "", "bold"),
        ("signature", "dark cyan,bold", "", "bold"),
        ("param", "dark blue", "", "bold"),
        ("section", "dark magenta,bold", "", "bold"),
        ("unknown", "white", "dark red", "bold"),
        ("directive", "white", "dark red", "bold"),
        # pygments
        ("pyg-o", "dark blue", "", "bold"),
        ("pyg-mi", "dark red", "", "bold"),
        ("pyg-kc", "dark green", "", "bold"),
        ("pyg-nb", "white", "", "bold"),
    ]

    # use appropriate Screen class
    if urwid.web_display.is_web_request():
        screen = urwid.web_display.Screen()
    else:
        screen = urwid.raw_display.Screen()

    def unhandled(key):
        if key == "q":
            raise urwid.ExitMainLoop()
        elif key == "backspace":
            if len(stack) >= 2:
                stack.pop()
                old = stack.pop()
                guess_load(old)

    urwid.MainLoop(frame, palette, screen, unhandled_input=unhandled).run()


def setup():
    urwid.web_display.set_preferences("Urwid Tour")
    # try to handle short web requests quickly
    if urwid.web_display.handle_short_request():
        return
    import sys

    target = sys.argv[1]
    main(target)


if "__main__" == __name__ or urwid.web_display.is_web_request():
    setup()
