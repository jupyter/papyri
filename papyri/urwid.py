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


def main():
    text_header = (
        "Welcome to the urwid tour!  "
        "UP / DOWN / PAGE UP / PAGE DOWN scroll.  F8 exits."
    )
    text_intro = [
        ("important", "Text", "Link for text"),
        " widgets are the most common in "
        "any urwid program.  This Text widget was created "
        "without setting the wrap or align mode, so it "
        "defaults to left alignment with wrapping on space "
        "characters.  ",
        ("bb", "Change the window width", "this is also clickable"),
        " to see how the widgets on this page react.  "
        "This Text widget is wrapped with a ",
        ("important", "Padding", "link for padding"),
        " widget to keep it indented on the left and right.",
    ]
    text_right = (
        "This Text widget is right aligned.  Wrapped "
        "words stay to the right as well. "
    )
    text_center = "This one is center aligned."
    text_clip = (
        "Text widgets may be clipped instead of wrapped.\n"
        "Extra text is discarded instead of wrapped to the next line. "
        "65-> 70-> 75-> 80-> 85-> 90-> 95-> 100>\n"
        "Newlines embedded in the string are still respected."
    )
    text_right_clip = (
        "This is a right aligned and clipped Text widget.\n"
        "<100 <-95 <-90 <-85 <-80 <-75 <-70 <-65             "
        "Text will be cut off at the left of this widget."
    )
    text_center_clip = (
        "Center aligned and clipped widgets will have " "text cut off both sides."
    )
    text_ellipsis = (
        "Text can be clipped using the ellipsis character (…)\n"
        "Extra text is discarded and a … mark is shown."
        "50-> 55-> 60-> 65-> 70-> 75-> 80-> 85-> 90-> 95-> 100>\n"
    )
    text_any = (
        "The 'any' wrap mode will wrap on any character.  This "
        "mode will not collapse space characters at the end of the "
        "line but it still honors embedded newline characters.\n"
        "Like this one."
    )
    text_padding = (
        "Padding widgets have many options.  This "
        "is a standard Text widget wrapped with a Padding widget "
        "with the alignment set to relative 20% and with its width "
        "fixed at 40."
    )
    text_divider = [
        "The ",
        ("important", "Divider"),
        " widget repeats the same character across the whole line.  "
        "It can also add blank lines above and below.",
    ]
    text_edit = [
        "The ",
        ("important", "Edit"),
        " widget is a simple text editing widget.  It supports cursor "
        "movement and tries to maintain the current column when focus "
        "moves to another edit widget.  It wraps and aligns the same "
        "way as Text widgets.",
    ]
    text_edit_cap1 = ("editcp", "This is a caption.  Edit here: ")
    text_edit_text1 = "editable stuff"
    text_edit_cap2 = ("editcp", "This one supports newlines: ")
    text_edit_text2 = (
        "line one starts them all\n"
        "== line 2 == with some more text to edit.. words.. whee..\n"
        "LINE III, the line to end lines one and two, unless you "
        "change something."
    )
    text_edit_cap3 = ("editcp", "This one is clipped, try " "editing past the edge: ")
    text_edit_text3 = "add some text here -> -> -> ...."
    text_edit_alignments = "Different Alignments:"
    text_edit_left = "left aligned (default)"
    text_edit_center = "center aligned"
    text_edit_right = "right aligned"
    text_intedit = ("editcp", [("important", "IntEdit"), " allows only numbers: "])
    text_edit_padding = ("editcp", "Edit widget within a Padding widget ")
    text_columns1 = [
        ("important", "Columns"),
        " are used to share horizontal screen space.  "
        "This one splits the space into two parts with "
        "three characters between each column.  The "
        "contents of each column is a single widget.",
    ]
    text_columns2 = [
        "When you need to put more than one " "widget into a column you can use a ",
        ("important", "Pile"),
        " to combine two or more widgets.",
    ]
    text_col_columns = "Columns may be placed inside other columns."
    text_col_21 = "Col 2.1"
    text_col_22 = "Col 2.2"
    text_col_23 = "Col 2.3"
    text_column_widths = (
        "Columns may also have uneven relative "
        "weights or fixed widths.  Use a minimum width so that "
        "columns don't become too small."
    )
    text_weight = "Weight %d"
    text_fixed_9 = "<Fixed 9>"  # should be 9 columns wide
    text_fixed_14 = "<--Fixed 14-->"  # should be 14 columns wide
    text_edit_col_cap1 = ("editcp", "Edit widget within Columns")
    text_edit_col_text1 = "here's\nsome\ninfo"
    text_edit_col_cap2 = ("editcp", "and within Pile ")
    text_edit_col_text2 = "more"
    text_edit_col_cap3 = ("editcp", "another ")
    text_edit_col_text3 = "still more"
    text_gridflow = [
        "A ",
        ("important", "GridFlow"),
        " widget "
        "may be used to display a list of flow widgets with equal "
        "widths.  Widgets that don't fit on the first line will "
        "flow to the next.  This is useful for small widgets that "
        "you want to keep together such as ",
        ("important", "Button"),
        ", ",
        ("important", "CheckBox"),
        " and ",
        ("important", "RadioButton"),
        " widgets.",
        "that's it",
    ]
    text_button_list = [
        "Yes",
        "No",
        "Perhaps",
        "Certainly",
        "Partially",
        "Tuesdays Only",
        "Help",
    ]
    text_cb_list = ["Wax", "Wash", "Buff", "Clear Coat", "Dry", "Racing Stripe"]
    text_rb_list = ["Morning", "Afternoon", "Evening", "Weekend"]
    text_listbox = [
        "All these widgets have been displayed " "with the help of a ",
        ("important", "ListBox"),
        " widget.  " "ListBox widgets handle scrolling and changing focus.  A ",
        ("important", "Frame"),
        " widget is used to keep the " "instructions at the top of the screen.",
    ]

    def button_press(button):
        frame.footer = urwid.AttrWrap(
            urwid.Text(["Pressed: ", button.get_label()]), "header"
        )


    radio_button_group = []

    blank = urwid.Divider()


    listbox_content_more = [
        Text("this is some text"),
        Text("And more text"),
        urwid.Button("Annother buttong", button_press),
        blank,
        Text(text_right, align="right"),
        blank,
        Text(text_center, align="center"),
        blank,
        Text(text_clip, wrap="clip"),
        blank,
        Text(text_right_clip, align="right", wrap="clip"),
        blank,
        Text(text_center_clip, align="center", wrap="clip"),
        blank,
        Text(text_ellipsis, wrap="ellipsis"),
        blank,
        Text(text_any, wrap="any"),
        blank,
        urwid.Padding(Text(text_padding), ("relative", 20), 40),
        blank,
        urwid.AttrWrap(urwid.Divider("=", 1), "bright"),
        urwid.Padding(Text(text_divider), left=2, right=2, min_width=20),
        urwid.AttrWrap(urwid.Divider("-", 0, 1), "bright"),
        blank,
        urwid.Padding(TextWithLink(text_edit), left=2, right=2, min_width=20),
        blank,
        urwid.AttrWrap(urwid.Edit(text_edit_cap1, text_edit_text1), "editbx", "editfc"),
        blank,
        urwid.AttrWrap(urwid.Edit(text_edit_cap2, text_edit_text2,
            multiline=True ), 'editbx', 'editfc'),
        blank,
        urwid.AttrWrap(
            urwid.Edit(text_edit_cap3, text_edit_text3, wrap="clip"), "editbx", "editfc"
        ),
        blank,
        Text(text_edit_alignments),
        urwid.AttrWrap(
            urwid.Edit("", text_edit_left, align="left"), "editbx", "editfc"
        ),
        urwid.AttrWrap(
            urwid.Edit("", text_edit_center, align="center"), "editbx", "editfc"
        ),
        urwid.AttrWrap(
            urwid.Edit("", text_edit_right, align="right"), "editbx", "editfc"
        ),
        blank,
        urwid.AttrWrap(urwid.IntEdit(text_intedit, 123),
            'editbx', 'editfc' ),
        blank,
        urwid.Padding(urwid.AttrWrap(urwid.Edit(text_edit_padding, ""),
            'editbx','editfc' ), left=10, width=50),
        blank,
        blank,
        urwid.AttrWrap(urwid.Columns([
            urwid.Divider("."),
            urwid.Divider(","),
            urwid.Divider("."),
            ]), 'bright'),
        blank,
        urwid.Columns(
            [
                urwid.Padding(Text(text_columns1), left=2, right=0, min_width=20),
                urwid.Pile(
                    [
                        urwid.Divider("~"),
                        Text(text_columns2),
                        urwid.Divider("_"),
                    ]
                ),
            ],
            3,
        ),
        blank,
        blank,
        urwid.Columns(
            [
                Text(text_col_columns),
                urwid.Columns(
                    [
                        Text(text_col_21),
                        Text(text_col_22),
                        Text(text_col_23),
                    ],
                    1,
                ),
            ],
            2,
        ),
        blank,
        urwid.Padding(Text(text_column_widths), left=2, right=2, min_width=20),
        blank,
        urwid.Columns(
            [
                urwid.AttrWrap(Text(text_weight % 1), "reverse"),
                ("weight", 2, Text(text_weight % 2)),
                ("weight", 3, urwid.AttrWrap(Text(text_weight % 3), "reverse")),
                ("weight", 4, Text(text_weight % 4)),
                ("weight", 5, urwid.AttrWrap(Text(text_weight % 5), "reverse")),
                ("weight", 6, Text(text_weight % 6)),
            ],
            0,
            min_width=8,
        ),
        blank,
        urwid.Columns(
            [
                ("weight", 2, urwid.AttrWrap(Text(text_weight % 2), "reverse")),
                ("fixed", 9, Text(text_fixed_9)),
                ("weight", 3, urwid.AttrWrap(Text(text_weight % 2), "reverse")),
                ("fixed", 14, Text(text_fixed_14)),
            ],
            0,
            min_width=8,
        ),
        blank,
        urwid.Columns([
            urwid.AttrWrap(urwid.Edit(text_edit_col_cap1,
                text_edit_col_text1, multiline=True),
                'editbx','editfc'),
            urwid.Pile([
                urwid.AttrWrap(urwid.Edit(
                    text_edit_col_cap2,
                    text_edit_col_text2),
                    'editbx','editfc'),
                blank,
                urwid.AttrWrap(urwid.Edit(
                    text_edit_col_cap3,
                    text_edit_col_text3),
                    'editbx','editfc'),
                ]),
            ], 1),
        blank,
        urwid.AttrWrap(
            urwid.Columns(
                [
                    urwid.Divider("'"),
                    urwid.Divider('"'),
                    urwid.Divider("~"),
                    urwid.Divider('"'),
                    urwid.Divider("'"),
                ]
            ),
            "bright",
        ),
        blank,
        blank,
        urwid.Padding(Text(text_gridflow), left=2, right=2, min_width=20),
        blank,
        urwid.Padding(
            urwid.GridFlow(
                [
                    urwid.AttrWrap(urwid.Button(txt, button_press), "buttn", "buttnf")
                    for txt in text_button_list
                ],
                13,
                3,
                1,
                "left",
            ),
            left=4,
            right=3,
            min_width=13,
        ),
        blank,
        urwid.Padding(urwid.GridFlow(
            [urwid.AttrWrap(urwid.CheckBox(txt),'buttn','buttnf')
                for txt in text_cb_list],
            10, 3, 1, 'left') ,
            left=4, right=3, min_width=10),
        blank,
        urwid.Padding(
            urwid.GridFlow(
                [
                    urwid.AttrWrap(
                        urwid.RadioButton(radio_button_group, txt), "buttn", "buttnf"
                    )
                    for txt in text_rb_list
                ],
                13,
                3,
                1,
                "left",
            ),
            left=4,
            right=3,
            min_width=13,
        ),
        blank,
        blank,
        urwid.Padding(Text(text_listbox), left=2, right=2, min_width=20),
        blank,
        blank,
    ]

    walk = urwid.SimpleListWalker([])

    def cb(value):
        def callback():
            assert isinstance(value, str)
            frame.footer = urwid.AttrWrap(urwid.Text(["Enter ?...: ", value]), "header")
            walk.append(listbox_content_more.pop(0))

        return callback

    acc = []

    def c1():
        walk.append(listbox_content_more.pop(0))

    def c2():
        listbox_content_more.append(walk.pop())

    text_intro = [
        "the widget ",
        ("important", "TextWithLink "),
        "allow inline text to be selcted, for example selecting ",
        ("important", "this text and pressing enter", c1),
        " will append more content" "to the list box, and pressing ",
        ("important", "this one will remove content", c2),
        " text does not need to be styled to be selectable and trigger callbacks:",
        (None, " > try to select this < ", c1),
        "Though it is recommended to add a style to links...",
    ]

    for t in text_intro:
        if isinstance(t, tuple) and len(t) == 3:
            ll = t[2]
            acc.append(Link(t[0], t[1], t[2]))
        else:
            acc.append(t)
    text_intro = acc

    listbox_content_extra = [
        urwid.Button("HEEEEY", button_press),
        blank,
        urwid.Padding(TextWithLink(text_intro), left=2, right=2, min_width=20),
    ]

    walk.extend(listbox_content_extra)

    header = urwid.AttrWrap(Text(text_header), "header")
    listbox = urwid.ListBox(walk)
    frame = urwid.Frame(urwid.AttrWrap(listbox, "body"), header=header)

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
    ]

    # use appropriate Screen class
    if urwid.web_display.is_web_request():
        screen = urwid.web_display.Screen()
    else:
        screen = urwid.raw_display.Screen()

    def unhandled(key):
        if key == 'f8':
            raise urwid.ExitMainLoop()

    urwid.MainLoop(frame, palette, screen,
        unhandled_input=unhandled).run()

def setup():
    urwid.web_display.set_preferences("Urwid Tour")
    # try to handle short web requests quickly
    if urwid.web_display.handle_short_request():
        return

    main()

if '__main__'==__name__ or urwid.web_display.is_web_request():
    setup()
