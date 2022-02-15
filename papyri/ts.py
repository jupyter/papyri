from pathlib import Path

from tree_sitter import Language, Parser

from . import errors
from .errors import (
    VisitCitationNotImplementedError,
    VisitCitationReferenceNotImplementedError,
    VisitSubstitutionDefinitionNotImplementedError,
)

pth = str(Path(__file__).parent / "rst.so")

RST = Language(pth, "rst")
parser = Parser()
parser.set_language(RST)

from textwrap import indent
from typing import List

from there import print

from papyri.take2 import (
    BlockDirective,
    BlockQuote,
    BlockVerbatim,
    BulletList,
    Comment,
    DefList,
    DefListItem,
    Directive,
    Emph,
    EnumeratedList,
    FieldList,
    FieldListItem,
    Options,
    Paragraph,
    Section,
    Strong,
    SubstitutionRef,
    Unimplemented,
    Verbatim,
    Word,
    Words,
    compress_word,
)


class Node:
    """
    A wrapper around tree sitter Nodes to slightly modify behavior.

    In particular we want to be able to extract whitespace information,
    which is made hard by tree sitter.

    So we intercept iterating through childrens, and if the bytes start/stop
    don't match, we insert a fake Whitespace node that has similar api to tree
    sitter official nodes.
    """

    def tree(self):
        return (
            repr(self)
            + indent("\n" + "\n".join([x.tree() for x in self.children]), "   ")
        ).rstrip()

    @property
    def children(self):
        if not self._with_whitespace:
            return [Node(n, _with_whitespace=False) for n in self.node.children]

        self.node.children
        current_byte = self.start_byte
        current_point = self.start_point
        new_nodes = []
        if self.node.children:
            for n in self.node.children:
                if n.start_byte != current_byte:
                    # value = self.bytes[current_byte:n.start_byte]
                    # assert value == b' ', (value, b' ', n)
                    new_nodes.append(
                        Whitespace(
                            current_byte, n.start_byte, current_point, n.start_point
                        )
                    )
                current_byte = n.end_byte
                current_point = n.end_point
                new_nodes.append(Node(n))

            if current_byte != self.end_byte:
                new_nodes.append(
                    Whitespace(
                        current_byte, self.end_byte, self.start_point, self.end_point
                    )
                )
        return new_nodes

    def __repr__(self):
        return repr(self.node)

    def with_whitespace(self):
        return Node(self.node, _with_whitespace=True)

    def without_whitespace(self):
        return Node(self.node, _with_whitespace=False)

    @property
    def start_point(self):
        return self.node.start_point

    @property
    def end_point(self):
        return self.node.end_point

    @property
    def start_byte(self):
        return self.node.start_byte

    @property
    def end_byte(self):
        return self.node.end_byte

    @property
    def type(self):
        return self.node.type

    @property
    def bytes(self):
        return self.node.bytes

    def __init__(self, node, *, _with_whitespace=True):
        self.node = node
        self._with_whitespace = _with_whitespace


class Whitespace(Node):
    def __init__(self, byte_start, byte_end, start_point, end_point):
        self._start_byte = byte_start
        self._end_byte = byte_end
        self._end_point = end_point
        self._start_point = start_point

    @property
    def start_point(self):
        return self._start_point

    @property
    def end_point(self):
        return self._end_point

    @property
    def children(self):
        return []

    @property
    def start_byte(self):
        return self._start_byte

    @property
    def end_byte(self):
        return self._end_byte

    def __repr__(self):
        return f'<Node kind="whitespace", start_point={self.start_point}, end_point={self.end_point}>'

    @property
    def type(self):
        return "whitespace"


class TSVisitor:
    """
    Tree sitter Visitor,

    Walk the tree sitter tree and convert each node into our kind of internal node.

    """

    def __init__(self, bytes, root):
        self.bytes = bytes
        self.root = root
        self.depth = 0

    def as_text(self, node):
        return self.bytes[node.start_byte : node.end_byte].decode()

    def visit_document(self, node):
        new_node = node.without_whitespace()
        items = self.visit(new_node)
        res = [x for x in items if not isinstance(x, Whitespace)]
        return res

    def visit(self, node):
        self.depth += 1
        acc = []
        prev_end = None
        for c in node.children:
            kind = c.type
            if kind == "::":
                if acc and isinstance(acc[-1], Word):
                    word = acc.pop()
                    acc.append(Word(word.value + "::"))
                # else:
                #    acc.append(Word("::"))
                continue
            if not hasattr(self, "visit_" + kind):
                raise ValueError(
                    f"visit_{kind} not found while visiting {node}::\n{self.bytes[c.start_byte: c.end_byte].decode()!r}"
                )
            meth = getattr(self, "visit_" + kind, self.generic_visit)
            acc.extend(meth(c, prev_end=prev_end))
            prev_end = c.end_point
        self.depth -= 1
        return acc

    def visit_citation(self, node, prev_end=None):
        # raise VisitCitationNotImplementedError()
        # just hlines, like ------
        return [Unimplemented("citation", self.as_text(node))]

    def visit_citation_reference(self, node, prev_end=None):
        raise VisitCitationReferenceNotImplementedError()
        # just hlines, like ------
        return []

    def visit_transition(self, node, prev_end=None):
        # assert False, self.bytes[node.start_byte - 10 : node.end_byte + 10].decode()
        data = self.bytes[node.start_byte : node.end_byte].decode()
        return [Paragraph([Words(data)], [])]

    def visit_reference(self, node, prev_end=None):
        """
        TODO:

        Currently we parse that as a directive, but actually it should be a reference
        and according to https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#hyperlink-references,
        there might be one or two trailing underscore, which we should pay attention to.


        """
        full_text = self.bytes[node.start_byte : node.end_byte].decode()
        if "`" not in full_text:
            # TODO reference do not need to be in backticks,
            # though it conflict with some things like numpy
            # direct references to np.bool_, np.complex64_ (see pandas docs for example)
            # we should likely have a way to handle that.
            _text = full_text
        else:
            _text, trailing = (
                self.bytes[node.start_byte + 1 : node.end_byte]
                .decode()
                .rsplit("`", maxsplit=1)
            )
            assert trailing in ("_", "__")
        return [Directive(_text, None, None)]

    def visit_interpreted_text(self, node, prev_end=None):
        if len(node.children) == 2:
            role, text = node.children
            assert role.type == "role"
            assert text.type == "interpreted_text"

            role_value = self.bytes[role.start_byte : role.end_byte].decode()
            assert role_value.startswith(":")
            assert role_value.endswith(":")
            role_value = role_value[1:-1]
            domain = None
            if ":" in role_value:
                domain, role_value = role_value.split(":")
                assert ":" not in role_value
                assert ":" not in domain

        elif len(node.children) == 1:
            [text] = node.children
            assert text.type == "interpreted_text"
            domain = None
            role = None
            role_value = None
        else:
            assert False

        text_value = self.bytes[text.start_byte : text.end_byte].decode()
        assert text_value.startswith("`")
        assert text_value.endswith("`")

        t = Directive(
            text_value[1:-1],
            domain=None,
            role=role_value,
        )
        return [t]

    def visit_standalone_hyperlink(self, node, prev_end=None):
        return self.visit_text(node)

    def visit_text(self, node, prev_end=None):
        t = Word(self.bytes[node.start_byte : node.end_byte].decode())
        t.start_byte = node.start_byte
        t.end_byte = node.end_byte
        # print(' '*self.depth*4, t, node.start_byte, node.end_byte)
        return [t]

    def visit_whitespace(self, node, prev_end=None):
        content = self.bytes[node.start_byte : node.end_byte].decode()
        # assert set(content) == {' '}, repr(content)
        t = Word(" " * len(content))
        t.start_byte = node.start_byte
        t.end_byte = node.end_byte
        # print(' '*self.depth*4, t, node.start_byte, node.end_byte)
        return [t]

    def visit_literal(self, node, prev_end=None):
        t = Verbatim([self.bytes[node.start_byte + 2 : node.end_byte - 2].decode()])
        # print(' '*self.depth*4, t)
        return [t]

    def visit_literal_block(self, node, prev_end=None):
        data = self.bytes[node.start_byte : node.end_byte].decode().splitlines()
        dedent_amount = node.start_point[1]

        # here we need to do a bit of custom logic to properly dedent
        acc = [data[0]]
        for x in data[1:]:
            # TODO : maybe assert here that what we remove is actually only whitespace ?
            # should we have a blcok verbatim with the first node more indented than subsequent ones ?
            acc.append(x[dedent_amount:])

        b = BlockVerbatim("\n".join(acc))

        # print(' '*self.depth*4, b)
        return [b]

    def visit_bullet_list(self, node, prev_end=None):
        acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
            assert len(list_item.children) == 2, list_item.children
            _bullet, body = list_item.children
            # assert len(body.children) == 1
            # parg = body.children[0]
            # assert parg.type == "paragraph", parg.type
            acc.extend(self.visit(body))
        return [BulletList(acc)]

        # t = Verbatim([self.bytes[node.start_byte+2: node.end_byte-2].decode()])
        # print(' '*self.depth*4, t)
        # return [t]

    def visit_section(self, node, prev_end=None):
        # print(' '*self.depth*4, '->', node)
        # print(' '*self.depth*4, '::',self.bytes[node.start_byte: node.end_byte].decode())
        if node.children[0].type == "adornment":
            assert node.children[1].type == "title"
            tc = node.children[1]
            assert node.children[2].type == "adornment"
            assert len(node.children) == 3
        else:
            assert node.children[0].type == "title"
            tc = node.children[0]
            assert node.children[1].type == "adornment"
            assert len(node.children) == 2
        title = self.bytes[tc.start_byte : tc.end_byte].decode()
        # print(' '*self.depth*4, '== Section: ', title, '==')
        # print(' '*self.depth*4, '->', node)
        return [Section([], title)]

    def visit_block_quote(self, node, prev_end=None):
        # print(indent(self.bytes[node.start_byte: node.end_byte].decode(), '> '))
        data = self.bytes[node.start_byte : node.end_byte].decode().splitlines()
        ded = node.start_point[1]
        acc = [data[0]]
        for x in data[1:]:
            acc.append(x[ded:])
        b = BlockQuote(acc)
        # print(' '*self.depth*4, b)
        return [b]

    def visit_paragraph(self, node, prev_end=None):
        sub = self.visit(node.with_whitespace())
        acc = []
        acc2 = []

        for item in sub:
            if isinstance(item, BlockVerbatim):
                acc2.append(item)
                continue
            acc.append(item)
        if acc[-1] == Word(" "):
            acc.pop()
        assert len(acc2) < 2
        p = Paragraph(compress_word(acc), [])
        # p.to_json()
        return [p, *acc2]

    def generic_visit(self, node):
        print("G" + " " * (self.depth * 4 - 1), node)
        assert False, node
        return []
        res = self.visit(node)
        # res.to_json()
        return res

    def visit_line_block(self, node, prev_end=None):
        # TODO
        return []

    def visit_substitution_reference(self, node, prev_end=None):
        # TODO
        return [SubstitutionRef(self.as_text(node))]

    def visit_doctest_block(self, node, prev_end=None) -> List[BlockVerbatim]:
        # TODO
        return self.visit_literal_block(node, prev_end)

    def visit_field(self, node, prev_end=None):
        return []

    def visit_field_list(self, node, prev_end=None):
        acc = []

        lens = {len(f.children) for f in node.children}
        if lens == {3}:
            # we likely have an option list
            for list_item in node.children:
                assert list_item.type == "field"
                _, name, _ = list_item.children
                # TODO, assert _ and _ are `:`
                acc.append(self.as_text(name))
            return [Options(acc)]

        elif lens == {4}:
            for list_item in node.children:
                assert list_item.type == "field"
                _, name, _, body = list_item.children
                a, b = compress_word(self.visit(name)), compress_word(self.visit(body))
                # [_.to_json()for _ in a]
                # [_.to_json() for _ in b]
                f = FieldListItem(a, b)
                acc.append(f)
            return [FieldList(acc)]
        else:
            raise ValueError("mixed len...")

    def visit_enumerated_list(self, node, prev_end=None):
        acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
            _bullet, body = list_item.children
            acc.extend(self.visit(body))
        return [EnumeratedList(acc)]

    def visit_target(self, node, prev_end=None):
        # TODO:
        # raise VisitTargetNotImplementedError()
        return [Unimplemented("target", self.as_text(node))]

    # def visit_arguments(self, node, prev_end=None):
    #    assert False
    #    return []

    def visit_inline_target(self, node, prev_end):
        # NotImplemented
        return [Unimplemented("inline_target", self.as_text(node))]

    def visit_directive(self, node, prev_end=None):
        # TODO:
        # make it part of the type if a block directive (has, or not), a body.

        # directive_name: str
        # args0: List[str]
        ## TODO : this is likely wrong...
        # inner: Optional[Paragraph]

        if len(node.children) == 4:
            _, _role, cc, body = node.children
            body_children = body.children
        elif len(node.children) == 3:
            _, _role, cc = node.children
            body_children = []
        else:
            raise ValueError

        if _role.end_point != cc.start_point:
            block_data = self.bytes[node.start_byte : node.end_byte].decode()
            raise errors.SpaceAfterBlockDirectiveError(
                f"space present in {block_data!r}"
            )

        role = self.bytes[_role.start_byte : _role.end_byte].decode()

        if len(body_children) == 2:
            arguments, content = body_children
            args0 = (
                self.bytes[
                    arguments.children[0].start_byte : arguments.children[-1].end_byte
                ]
                .decode()
                .splitlines()
            )
        else:
            if len(body_children) == 0:
                pass
            elif len(body_children) > 1:
                content = [c for c in body_children if c.type == "content"]
                content = content[0]
                import warnings

                warnings.warn("TBD directive arguments")
            else:
                [content] = body_children
            args0 = []
        if len(body_children) == 0:
            stream_with_spaces = []
        else:
            stream_with_spaces = [
                x for y in [(x, Word(" ")) for x in self.visit(content)] for x in y
            ]
        directive = BlockDirective(
            directive_name=role,
            args0=args0,
            inner=Paragraph(compress_word(stream_with_spaces), []),
        )
        return [directive]

    def visit_footnote_reference(self, node, prev_end=None):
        # TODO
        # assert False, self.bytes[node.start_byte : node.end_byte].decode()
        return []

    def visit_emphasis(self, node, prev_end=None):
        # TODO
        return [
            Emph(Words(self.bytes[node.start_byte + 1 : node.end_byte - 1].decode()))
        ]

    def visit_substitution_definition(self, node, prev_end=None):
        # TODO
        raise VisitSubstitutionDefinitionNotImplementedError(
            self.bytes[node.start_byte : node.end_byte].decode()
        )
        return []

    def visit_comment(self, node, prev_end=None):
        # TODO
        return [Comment(self.bytes[node.start_byte : node.end_byte].decode())]
        # raise VisitCommentNotImplementedError()

    def visit_strong(self, node, prev_end=None):
        return [
            Strong(Words(self.bytes[node.start_byte + 2 : node.end_byte - 2].decode()))
        ]

    def visit_footnote(self, node, prev_end=None):
        # TODO
        # that is actually used for references
        # assert False, self.bytes[node.start_byte : node.end_byte].decode()
        return [Unimplemented("footnote", self.as_text(node))]

    def visit_ERROR(self, node, prev_end=None):
        """
        Called with parsing error nodes.
        """
        # TODO
        # raise TreeSitterParseError()
        return []

    def visit_definition_list(self, node, prev_end=None):

        acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
            if len(list_item.children) == 2:
                term, definition = list_item.children
                assert term.type == "term"
                assert definition.type == "definition"
                # dl = Paragraph(
                #     [Words(self.bytes[term.start_byte : term.end_byte].decode())], []
                # )
                _dd = self.visit(definition)
                acc.append(
                    DefListItem(
                        dt=self.visit_paragraph(term)[0],
                        dd=_dd,
                    )
                )
            elif len(list_item.children) == 4:
                term, _, classsifier, definition = list_item.children
                assert term.type == "term"
                assert definition.type == "definition"
                assert classsifier.type == "classifier"
                assert _.type == ":"
                _dd = self.visit(definition)
                # TODO missing type
                acc.append(
                    DefListItem(
                        dt=Paragraph(compress_word(self.visit(term)), []),
                        dd=self.visit_paragraph(term),
                    )
                )
            else:
                # TODO
                return []
                assert False, list_item.children

        return [DefList(acc)]


def nest_sections(items) -> List[Section]:
    if not items:
        return []
    acc = []
    if not isinstance(items[0], Section):
        acc.append(Section([]))
    for item in items:
        if isinstance(item, Section):
            acc.append(item)
        else:
            acc[-1].children.append(item)
    return acc


def parse(text: bytes) -> List[Section]:
    """
    Parse text using Tree sitter RST, and return a list of serialised section I guess ?
    """

    tree = parser.parse(text)
    root = Node(tree.root_node)
    return nest_sections(TSVisitor(text, root).visit_document(root))


class TreeSitterParseError(Exception):
    pass
