"""
An attempt to create AST from MyST spec. At the moment just trying to
add Text from Myst to the generated structure in the ts.py.

"""
from typing import List, Union, Any

from papyri.common_ast import Node


# Prefixed with "M" as Text exists in take2.py as well
class MText(Node):
    type = 'text'
    value: str
    # position: Any
    # data: Any


class MEmphasis(Node):
    type = "emphasis"
    children: List['PhrasingContent']
    # position: Any
    # data: Any


class MStrong(Node):
    type = "strong"
    # position: Any
    # data: Any


class MLink(Node):
    type = 'link'
    children: List['StaticPhrasingContent']
    url: Any
    title: Any
    # position: Any
    # data: Any


class MCode(Node):
    type = 'code'
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


class MParagraph(Node):
    type = 'paragraph'
    children: List['PhrasingContent']
    # position: Any
    # data: Any


StaticPhrasingContent = Union[
    MText,
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
