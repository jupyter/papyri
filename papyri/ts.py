from pathlib import Path

from tree_sitter import Language, Parser

pth = str(Path(__file__).parent / "rst.so")

RST = Language(pth, "rst")
parser = Parser()
parser.set_language(RST)

from typing import List

from papyri.take2 import (
    BlockDirective,
    BlockQuote,
    BlockVerbatim,
    BulletList,
    DefList,
    DefListItem,
    Directive,
    EnumeratedList,
    FieldList,
    FieldListItem,
    Lines,
    Paragraph,
    Section,
    Verbatim,
    Word,
    Words,
    compress_word,
)


class TSVisitor:
    """
    Tree sitter Visitor,

    Walk the tree sitter tree and convert each node into our kind of internal node.

    """

    def __init__(self, bytes, root):
        self.bytes = bytes
        self.root = root
        self.depth = 0

    def visit(self, node):
        self.depth += 1
        acc = []
        prev_end = None
        for c in node.children:
            kind = c.type
            # print(f'({kind})')
            if kind == "::":
                if isinstance(acc[-1], Word):
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
        assert False
        # just hlines, like ------
        return []

    def visit_citation_reference(self, node, prev_end=None):
        assert False
        # just hlines, like ------
        return []

    def visit_transition(self, node, prev_end=None):
        # assert False, self.bytes[node.start_byte - 10 : node.end_byte + 10].decode()
        data = self.bytes[node.start_byte : node.end_byte].decode()
        return [Paragraph([Words(data)], [])]

    def visit_reference(self, node, prev_end=None):
        t = Directive(
            [self.bytes[node.start_byte : node.end_byte - 1].decode()], None, None
        )
        # print(' '*self.depth*4, t)
        return [t]

    def visit_interpreted_text(self, node, prev_end=None):
        t = Directive(
            [self.bytes[node.start_byte + 1 : node.end_byte - 1].decode()], None, None
        )
        # print(' '*self.depth*4, t)
        return [t]

    def visit_standalone_hyperlink(self, node, prev_end=None):
        return self.visit_text(node)

    def visit_text(self, node, prev_end=None):
        t = Word(self.bytes[node.start_byte : node.end_byte].decode())
        # print(' '*self.depth*4, t, node.start_byte, node.end_byte)
        return [t]

    def visit_literal(self, node, prev_end=None):
        t = Verbatim([self.bytes[node.start_byte + 2 : node.end_byte - 2].decode()])
        # print(' '*self.depth*4, t)
        return [t]

    def visit_literal_block(self, node, prev_end=None):
        data = self.bytes[node.start_byte : node.end_byte].decode().splitlines()
        ded = node.start_point[1]
        acc = [data[0]]
        for x in data[1:]:
            acc.append(x[ded:])
        lines = Lines(acc)
        if prev_end is not None:
            for l in lines:
                l._number += node.start_point[0] - prev_end[0]

        b = BlockVerbatim(lines)

        # print(' '*self.depth*4, b)
        return [b]

    def visit_bullet_list(self, node, prev_end=None):
        acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
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
        sub = self.visit(node)
        # if Word("Example") in sub:
        # import ipdb

        # ipdb.set_trace()
        # sub = self.visit(node)
        # print("SUB,", sub, node)
        acc = []
        acc2 = []
        for item in sub:
            if isinstance(item, BlockVerbatim):
                acc2.append(item)
                continue
            acc.append(item)
            acc.append(Word(" "))
        if acc[-1] == Word(" "):
            acc.pop()
        assert len(acc2) < 2
        p = Paragraph(compress_word(acc), [])
        p.to_json()
        return [p, *acc2]

    def generic_visit(self, node):
        print("G" + " " * (self.depth * 4 - 1), node)
        assert False, node
        return []
        return self.visit(node)

    def visit_line_block(self, node, prev_end=None):
        # TODO
        assert False
        return []

    def visit_substitution_reference(self, node, prev_end=None):
        # TODO
        assert False
        return []

    def visit_doctest_block(self, node, prev_end=None) -> List[BlockVerbatim]:
        # TODO
        # likely want to dispatch to the parse example routine.
        return [
            BlockVerbatim(
                Lines(self.bytes[node.start_byte : node.end_byte].decode().splitlines())
            )
        ]

    def visit_field(self, node, prev_end=None):
        return []

    def visit_field_list(self, node, prev_end=None):
        acc = []
        for list_item in node.children:
            assert list_item.type == "field"
            _, name, _, body = list_item.children
            f = FieldListItem(self.visit(name), self.visit(body))
            acc.append(f)
        return [FieldList(acc)]
        return []

    def visit_enumerated_list(self, node, prev_end=None):
        acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
            _bullet, body = list_item.children
            acc.extend(self.visit(body))
        return [EnumeratedList(acc)]

    def visit_target(self, node, prev_end=None):
        # TODO:
        assert False
        return []

    # def visit_arguments(self, node, prev_end=None):
    #    assert False
    #    return []

    def visit_directive(self, node, prev_end=None):

        # directive_name: str
        # args0: List[str]
        ## TODO : this is likely wrong...
        # inner: Optional[Paragraph]

        _, _role, _, body = node.children
        role = self.bytes[_role.start_byte : _role.end_byte].decode()

        if len(body.children) == 2:
            arguments, content = body.children
            args0 = (
                self.bytes[
                    arguments.children[0].start_byte : arguments.children[-1].end_byte
                ]
                .decode()
                .splitlines()
            )
        else:
            [content] = body.children
            args0 = []

        #        import ipdb
        #
        #        ipdb.set_trace()

        directive = BlockDirective()
        directive.args0 = args0
        directive.directive_name = role
        stream = self.visit(content)
        stream_with_spaces = [
            x for y in [(x, Word(" ")) for x in self.visit(content)] for x in y
        ]
        directive.inner = Paragraph(compress_word(stream_with_spaces), [])
        directive.lines = Lines()
        directive.wh = Lines()
        directive.ind = Lines()
        return [directive]

    def visit_footnote_reference(self, node, prev_end=None):
        # TODO
        # assert False, self.bytes[node.start_byte : node.end_byte].decode()
        return []

    def visit_emphasis(self, node, prev_end=None):
        # TODO
        assert False
        return []

    def visit_substitution_definition(self, node, prev_end=None):
        # TODO
        assert False
        return []

    def visit_comment(self, node, prev_end=None):
        # TODO
        assert False
        return []

    def visit_strong(self, node, prev_end=None):
        # TODO
        assert False
        return []

    def visit_footnote(self, node, prev_end=None):
        # TODO
        # that is actually used for references
        # assert False, self.bytes[node.start_byte : node.end_byte].decode()
        return []

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
            # print(list_item.children)
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
                        Lines(),
                        Lines(),
                        Lines(),
                        dl=self.visit_paragraph(term)[0],
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
                        Lines(),
                        Lines(),
                        Lines(),
                        dl=Paragraph(compress_word(self.visit(term)), []),
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
    # just validation
    # that it's serialisable
    # no side effects.
    for a in acc:
        a.to_json()
    return acc


def parse(text: bytes) -> List[Section]:
    """
    Parse text using Tree sitter RST, and return a list of serialised section I guess ?
    """

    tree = parser.parse(text)
    return nest_sections(TSVisitor(text, tree.root_node).visit(tree.root_node))


class TreeSitterParseError(Exception):
    pass
