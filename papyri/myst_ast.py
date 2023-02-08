"""
An attempt to create AST from MyST spec. At the moment just trying to
add Text from Myst to the generated structure in the ts.py.

"""
from typing import List, Union, Any

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
    # position: Any
    # data: Any


@register(4049)
class MLink(Node):
    type = "link"
    children: List["StaticPhrasingContent"]
    url: Any
    title: Any
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


StaticPhrasingContent = Union[
    MText,
    MInlineCode,
    take2.Verbatim,
    take2.Directive,
    take2.Link,
    take2.Strong,
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
    take2.ExternalLink,
]

FlowContent = Union[
    MCode,
    MParagraph,
    # MDefinition,
    # MHeading,
    # MThematicBreak,
    # MBlockquote,
    # MList,
    # MHTML,
    # MComment,
    # MTarget,
    # MDirective,
    # MAdmonition,
    # MContainer,
    # MMath,
    # MTable,
    # MFootnoteDefinition,
]
