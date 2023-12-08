"""
Attempt to render using the rich protocol.
"""

from .myst_ast import MThematicBreak, MHeading
from dataclasses import dataclass
from rich.segment import Segment
from typing import Any, Optional
from rich.console import Console, ConsoleOptions, RenderResult, Group
from rich.panel import Panel, Box
from rich.padding import Padding
from rich import print
import rich
import json


def pad(arg):
    return Padding(arg, (0, 0, 0, 2))


def part(value: str, needle: str):
    while value:
        if needle in value:
            a, b, new_value = value.partition(needle)
            assert len(new_value) <= len(value)
            yield a
            yield b
            value = new_value
        else:
            break
    yield value


@dataclass
class RToken:
    value: str
    style: Optional[str] = None

    def __init__(self, value, style=None):
        self.value = value
        self.style = style

    def __len__(self):
        return len(self.value)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if self.style:
            yield Segment(self.value, console.get_style(self.style))
        else:
            yield Segment(self.value)

    def partition(self, needle=" "):
        return [RToken(c, self.style) for c in part(self.value, needle)]


@dataclass
class Unimp(RToken):
    style: str = "unimp"


@dataclass
class RTokenList:
    children: tuple[Any]

    def __add__(self, other):
        assert type(self) == type(other)

        return RTokenList(self.children + other.children)

    def __init__(self, children):
        for c in children:
            assert isinstance(c, (RToken, Unimp)), c
        self.children = children

    @classmethod
    def from_str(cls, value):
        return RTokenList([RToken(x) for x in part(value, " ")])

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        acc = 0
        options.max_width
        # TODO:, on newline eat whitespace
        for s in self.children:
            acc += len(s)
            assert isinstance(s, RToken)
            if acc >= options.max_width:
                if not s.value == " ":
                    acc = len(s)
                    yield Segment.line()
                    yield s
                    # yield str(acc)
                else:
                    acc = 0
                    yield Segment.line()
            else:
                yield s
                # yield Segment(str(acc) + "|")


DEBUG = False


@dataclass
class RichBlocks:
    children: list[Any]
    name: str

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        for item in self.children:
            if DEBUG:
                yield Panel(item, title=self.name)
            else:
                yield item
            # yield RToken("\n")


class RichVisitor:
    def visit(self, node):
        res = self.generic_visit([node])
        assert len(res) in (1, 0)
        if res:
            return res[0]
        else:
            return []

    def generic_visit(self, nodes):
        acc = []

        for node in nodes:
            name = node.__class__.__name__
            meth = getattr(self, f"visit_{name}")
            acc.extend(meth(node))

        return acc

    def visit_MRoot(self, node):
        cs = self.generic_visit(node.children)
        return [RichBlocks(cs, "root")]

    def visit_MParagraph(self, node):
        cs = self.generic_visit(node.children)
        return [RTokenList(cs) + RTokenList.from_str("\n\n")]

    def visit_MText(self, node):
        res = [RToken(v) for v in part(node.value, " ")]
        assert res[-1].value != "\n"
        return res

    def visit_MEmphasis(self, node):
        return self.generic_visit(node.children)

    def visit_MInlineCode(self, node):
        return RToken(node.value, "m.inline_code").partition()

    def visit_MList(self, node):
        return [pad(RichBlocks(self.generic_visit(node.children), "mlist"))]

    def visit_DefList(self, node):
        return [pad(RichBlocks(self.generic_visit(node.children), "deflist"))]

    def visit_MListItem(self, node):
        res = self.generic_visit(node.children)
        if len(res) == 1 and hasattr(res[0], "children"):
            return [RTokenList([RToken("- ")] + res[0].children)]
        else:
            return [RTokenList([RToken("- ")])] + res

    def visit_Directive(self, node):
        if node.domain:
            assert node.role
        content = ""
        if node.domain:
            content += f":{node.domain}"
        if node.role:
            content += f":{node.role}:"
        content += f"`{node.value}`"

        return RToken(content, "m.directive").partition()

    def visit_MLink(self, node):
        return self.generic_visit(node.children)

    def visit_MAdmonitionTitle(self, node):
        return [Panel(Unimp(str(node.to_dict())))]
        return self.generic_visit(node.children)

    def visit_MAdmonition(self, node):
        title, *other = node.children
        assert title.type == "admonitionTitle"
        acc = self.generic_visit([title])
        if other:
            rd = self.generic_visit(other)

            acc.append(Panel(Group(*rd)))
        return acc

    def visit_MHeading(self, node: MHeading):
        cs = [RToken("#" * (node.depth + 1) + " ")] + self.generic_visit(node.children)
        return [RTokenList(cs) + RTokenList.from_str("\n\n")]

    def visit_Param(self, node):
        cs = [
            RToken(node.param, "param"),
            RToken(" : "),
            RToken(node.type_, "param_type"),
            RToken("\n"),
        ]
        sub = self.generic_visit(node.desc)
        return [RTokenList(cs), pad(Group(*sub))]

    def visit_MMath(self, node):
        # maybe look like:
        # from sympy.parsing.latex import parse_latex
        # from sympy import pretty
        # pretty(parse_latex(...))
        from flatlatex import converter

        flat = converter().convert(node.value)

        formula = RToken(flat, style="math")
        return [Padding(formula, (1, 1, 1, 1))]

    def visit_MInlineMath(self, node):
        from flatlatex import converter

        flat = converter().convert(node.value)

        return [RToken(flat, style="math")]

    # return self.visit_unknown(node)

    def visit_FieldList(self, node):
        return self.visit_unknown(node)

    def visit_DefListItem(self, node):
        return self.generic_visit([node.dt]) + [
            pad(Group(*self.generic_visit(node.dd)))
        ]

    def visit_MCode(self, node):
        return [pad(Panel(node.value, expand=True, title="Code", highlight=True))]

    def visit_MBlockquote(self, node):
        sub = self.generic_visit(node.children)
        return [pad(Group(*sub))]

    def visit_unknown(self, node):
        print("Unimplemented node", node)
        return [Unimp(json.dumps(node.to_dict(), indent=2))]

    def visit_Parameters(self, node):
        return self.generic_visit(node.children)

    def visit_Unimplemented(self, node):
        return self.visit_unknown(node)

    def visit_SubstitutionDef(self, node):
        return self.visit_unknown(node)

    def visit_SubstitutionRef(self, node):
        return self.visit_unknown(node)

    def visit_MMystDirective(self, node):
        return self.visit_unknown(node)

    def visit_MStrong(self, node):
        return self.visit_unknown(node)

    def visit_MComment(self, node):
        return self.visit_unknown(node)

        return self.visit_unknown(node)

    def visit_MThematicBreak(self, node: MThematicBreak):
        return [rich.rule.Rule()]

    def visit_MUnimpl(self, node):
        return self.generic_visit(node.children)


xbox = Box(
    """
┌─┬┐
│ ││
├─┼┤
│ ││
├─┼┤
├─┼┤
│ ││
└─┴┘
""".strip(
        "\n"
    )
)


if __name__ == "__main__":
    print(
        Panel(
            Panel(
                RTokenList.from_str(
                    "Here is a long sentence to see if we can wrap " * 10
                ),
                box=xbox,
            ),
        )
    )
