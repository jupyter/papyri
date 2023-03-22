"""
An attempt to create AST from MyST spec. At the moment just trying to
add Text from Myst to the generated structure in the ts.py.

"""
from typing import List, Union, Optional, Dict

from .common_ast import Node, register

from . import take2


# Prefixed with "M" as Text exists in take2.py as well
@register(4046)
class MText(Node):
    type = "text"
    value: str
    # position: Any
    # data: Any


@register(4047)
class MEmphasis(Node):
    type = "emphasis"
    children: List["PhrasingContent"]
    # position: Any
    # data: Any


@register(4048)
class MStrong(Node):
    type = "strong"
    children: List["PhrasingContent"]
    # position: Any
    # data: Any


@register(4049)
class MLink(Node):
    type = "link"
    children: List["StaticPhrasingContent"]
    url: str
    title: str
    # position: Any
    # data: Any


@register(4050)
class MCode(Node):
    type = "code"
    # lang: str
    # meta: str
    # class_: str
    # show_line_numbers: bool
    # starting_line_number: int
    # emphasize_lines: List[int]
    # identifier: Any
    # label: Any
    value: str
    # position: Any
    # data: Any


@register(4051)
class MInlineCode(Node):
    type = "inlineCode"
    value: str
    # position: Any
    # data: Any


# class LinkReference:
#     type = 'linkReference'
#     children: List[StaticPhrasingContent]
#     referenceType: Any
#     identifier: Any
#     label: Any
# position: Any
# data: Any


# class Abbreviation:
#     type = "abbreviation"
#
#
# class CrossReference:
#     type = "crossReference"
#
#
# class FootnoteReference:
#     type = "footnoteReference"


@register(4045)
class MParagraph(Node):
    type = "paragraph"
    children: List["PhrasingContent"]
    # position: Any
    # data: Any


@register(4053)
class MList(Node):
    type = "list"
    ordered: bool
    # "Starting number of ordered list"
    start: int
    # "One or more children are separated with a blank line from others"
    spread: bool
    # "List of children nodes"
    # {"type": "array", "items": {"$ref": "#/$defs/Node"}}
    children: List["ListContent"]


@register(4054)
class MListItem(Node):
    type = "listItem"
    # "One or more children are separated with a blank line from others"
    spread: bool
    # "List of children nodes"
    children: List[
        Union[
            "FlowContent",
            "PhrasingContent",
            "take2.DefList",
            "take2.Paragraph",
            "take2.Admonition",
            "take2.BlockDirective",
            "take2.BlockVerbatim",
            "take2.BlockMath",
        ]
    ]


@register(4052)
class MMystDirective(Node):
    type = "mystDirective"
    name: str
    args: Optional[str]
    options: Dict[str, str]
    value: Optional[str]
    children: List[Union["FlowContent", "PhrasingContent", None]] = []


@register(4055)
class MAdmonitionTitle(Node):
    type = "admonitionTitle"
    children: List[Union["PhrasingContent", None]] = []


@register(4056)
class MAdmonition(Node):
    type = "admonition"
    children: List[
        Union[
            "FlowContent",
            "MAdmonitionTitle",
            take2.Unimplemented,
            "take2.DefList",
        ]
    ] = []
    kind: str = "note"


@register(4060)
class MComment(Node):
    type = "mystComment"
    value: str
    # position: Any
    # data: Any


@register(4058)
class MMath(Node):
    type = "math"
    value: str


@register(4057)
class MInlineMath(Node):
    type = "inlineMath"
    value: str
    # position: Any
    # data: Any


@register(4059)
class MBlockquote(Node):
    type = "blockquote"
    children: List["FlowContent"] = []
    # position: Any
    # data: Any


StaticPhrasingContent = Union[
    MText,
    MInlineCode,
    MInlineMath,
    take2.Verbatim,
    take2.Directive,
    take2.Link,
    take2.Math,
    take2.SubstitutionRef,
    take2.Unimplemented,
]

PhrasingContent = Union[
    StaticPhrasingContent,
    MEmphasis,
    MStrong,
    MLink,
    # LinkReference,
    # Subscript,
    # Superscript,
    # Underline,
    # Abbreviation,
    # CrossReference,
    # FootnoteReference,
]

FlowContent = Union[
    MCode,
    MParagraph,
    # MDefinition,
    # MHeading,
    # MThematicBreak,
    MBlockquote,
    MList,
    # MHTML,
    # MComment,
    # MTarget,
    MMystDirective,
    MAdmonition,
    # MContainer,
    MMath,
    "take2.DefList",
    "take2.DefListItem",
    # MTable,
    # MFootnoteDefinition,
]

ListContent = Union[MListItem,]
