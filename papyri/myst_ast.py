"""
An attempt to create AST from MyST spec.
"""

from typing import List, Union, Optional, Dict

from .common_ast import Node, register

from . import take2
from . import signature


# Prefixed with "M" as Text exists in take2.py as well
@register(4046)
class MText(Node):
    type = "text"
    value: str
    # position: Any
    # data: Any

    def __init__(self, value):
        assert isinstance(value, str)
        # assert not value.startswith(":func:")
        self.value = value
        super().__init__()


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

    def __init__(self, value):
        super().__init__(value)
        assert "\n" not in value


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
    children: List[Union["PhrasingContent", "take2.MUnimpl", "MImage"]]
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
            "UnprocessedDirective",
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

    @classmethod
    def from_unprocessed(cls, up):
        return cls(up.name, up.args, up.options, up.value, up.children)


class UnserializableNode(Node):
    _dont_serialise = True

    def cbor(self, encoder):
        assert False

    def to_json(self) -> bytes:
        assert False


class UnprocessedDirective(UnserializableNode):
    """
    Placeholder for yet unprocessed directives,  after they are parsed by tree-sitter,
    But before they are dispatched through the role resolution.
    """

    name: str
    args: Optional[str]
    options: Dict[str, str]
    value: Optional[str]
    children: List[Union["FlowContent", "PhrasingContent", None]]
    raw: str


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


@register(4061)
class MTarget(Node):
    type = "mystTarget"
    label: str
    # position: Any
    # data: Any


@register(4062)
class MImage(Node):
    type = "image"
    url: str
    alt: str


@register(4019)
class MThematicBreak(Node):
    type = "thematicBreak"


@register(4020)
class MHeading(Node):
    type = "heading"
    depth: int
    children: List["PhrasingContent"]


@register(4001)
class MRoot(Node):
    type = "root"
    children: List[
        Union[
            "FlowContent",
            "take2.Parameters",
            "take2.Unimplemented",
            "take2.SubstitutionDef",
            "signature.SignatureNode",
            MImage,
        ]
    ]


class ReplaceNode(Node):
    # We may want to return links too.
    type = "replace"
    value: str
    text: str
    # children: Union[
    #     MText,
    #     MInlineCode,
    #     MInlineMath,
    # ]


StaticPhrasingContent = Union[
    MText,
    MInlineCode,
    MInlineMath,
    take2.Directive,
    take2.Link,
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
    "UnprocessedDirective",
    MHeading,
    MThematicBreak,
    MBlockquote,
    MList,
    # MHTML,
    # MComment,
    MTarget,
    MMystDirective,
    MAdmonition,
    # MContainer,
    MMath,
    "take2.DefList",
    "take2.DefListItem",
    "take2.FieldList",
    MComment,  # this seem not right but triggers an error with numpy narrative documentation
    # MTable,
    # MFootnoteDefinition,
]

ListContent = Union[MListItem,]
