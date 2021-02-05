from tree_sitter import Language, Parser

RST = Language("/Users/bussonniermatthias/dev/papyri/build/rst.so", "rst")
parser = Parser()
parser.set_language(RST)

from textwrap import indent

from papyri.take2 import (
    Text,
    Verbatim,
    Paragraph,
    compress_word,
    Word,
    BlockQuote,
    Section,
    BulletList,
    DefList,
    DefListItem,
    Lines,
    Words,
    Directive,
    BlockVerbatim,
)


class TSW:
    def __init__(self, bytes, root):
        self.bytes = bytes
        self.root = root
        self.depth = 0

    def visit(self, node):

        self.depth += 1
        acc = []
        for c in node.children:
            kind = c.type
            # print(f'({kind})')
            if kind == "::":
                continue
            if not hasattr(self, "visit_" + kind):
                raise ValueError(
                    f"visit_{kind} not found while visiting {node}::\n{self.bytes[c.start_byte: c.end_byte].decode()!r}"
                )
            meth = getattr(self, "visit_" + kind, self.generic_visit)
            acc.extend(meth(c))
        self.depth -= 1
        return acc

    def visit_transition(self, node):
        # just hlines, like ------
        return []

    def visit_reference(self, node):
        t = Directive(
            [self.bytes[node.start_byte : node.end_byte - 1].decode()], None, None
        )
        # print(' '*self.depth*4, t)
        return [t]

    def visit_interpreted_text(self, node):
        t = Directive(
            [self.bytes[node.start_byte + 1 : node.end_byte - 1].decode()], None, None
        )
        # print(' '*self.depth*4, t)
        return [t]

    def visit_standalone_hyperlink(self, node):
        return self.visit_text(node)

    def visit_text(self, node):
        t = Word(self.bytes[node.start_byte : node.end_byte].decode())
        # print(' '*self.depth*4, t, node.start_byte, node.end_byte)
        return [t]

    def visit_literal(self, node):
        t = Verbatim([self.bytes[node.start_byte + 2 : node.end_byte - 2].decode()])
        # print(' '*self.depth*4, t)
        return [t]

    def visit_literal_block(self, node):
        data = self.bytes[node.start_byte : node.end_byte].decode().splitlines()
        ded = node.start_point[1]
        acc = [data[0]]
        for x in data[1:]:
            acc.append(x[ded:])
        b = BlockVerbatim(Lines(acc))
        # print(' '*self.depth*4, b)
        return [b]

    def visit_bullet_list(self, node):
        acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
            _bullet, body = list_item.children
            assert len(body.children) == 1
            parg = body.children[0]
            assert parg.type == "paragraph", parg.type
            acc.extend(self.visit(body))
        return [BulletList(acc)]

        # t = Verbatim([self.bytes[node.start_byte+2: node.end_byte-2].decode()])
        # print(' '*self.depth*4, t)
        # return [t]

    def visit_section(self, node):
        # print(' '*self.depth*4, '->', node)
        # print(' '*self.depth*4, '::',self.bytes[node.start_byte: node.end_byte].decode())
        assert node.children[0].type == "title"
        tc = node.children[0]
        title = self.bytes[tc.start_byte : tc.end_byte].decode()
        assert node.children[1].type == "adornment"
        assert len(node.children) == 2
        # print(' '*self.depth*4, '== Section: ', title, '==')
        # print(' '*self.depth*4, '->', node)
        return [Section([], title)]

    def visit_block_quote(self, node):
        # print(indent(self.bytes[node.start_byte: node.end_byte].decode(), '> '))
        data = self.bytes[node.start_byte : node.end_byte].decode().splitlines()
        ded = node.start_point[1]
        acc = [data[0]]
        for x in data[1:]:
            acc.append(x[ded:])
        b = BlockQuote(acc)
        # print(' '*self.depth*4, b)
        return [b]

    def visit_paragraph(self, node):
        sub = self.visit(node)
        acc = []
        acc2 = []
        for item in sub:
            if isinstance(item, BlockVerbatim):
                acc2.append(item)
                continue
            acc.append(item)
            acc.append(Word(" "))
        assert len(acc2) < 2
        p = Paragraph(compress_word(acc), acc2)
        p.to_json()
        # print(' '*self.depth*4, '-- Paragraph --')
        # print(indent(self.bytes[node.start_byte: node.end_byte].decode(), '| '))
        return [p]

    def generic_visit(self, node):
        print("G" + " " * (self.depth * 4 - 1), node)
        assert False, node
        return []
        return self.visit(node)

    def visit_doctest_block(self, node):
        # TODO
        return []

    def visit_field_list(self, node):
        # TODO
        return []

    def visit_enumerated_list(self, node):
        # TODO
        return []

    def visit_directive(self, node):
        # TODO
        return []

    def visit_definition_list(self, node):

        acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
            # print(list_item.children)
            if len(list_item.children) == 2:
                term, definition = list_item.children
                assert term.type == "term"
                assert definition.type == "definition"
                dl = Paragraph(
                    [Words(self.bytes[term.start_byte : term.end_byte].decode())], []
                )
                _dd = self.visit(definition)
                assert len(_dd) == 1, self.bytes[node.start_byte : node.end_byte]
                acc.append(
                    DefListItem(
                        Lines(),
                        Lines(),
                        Lines(),
                        dl=self.visit_paragraph(term)[0],
                        dd=_dd[0],
                    )
                )
            elif len(list_item.children) == 4:
                term, _, classsifier, definition = list_item.children
                assert term.type == "term"
                assert definition.type == "definition"
                assert classsifier.type == "classifier"
                assert _.type == ":"
                _dd = self.visit(definition)
                assert len(_dd) == 1
                # TODO missing type
                acc.append(
                    DefListItem(
                        Lines(),
                        Lines(),
                        Lines(),
                        dl=Paragraph(compress_word(self.visit(term)), []),
                        dd=self.visit_paragraph(term)[0],
                    )
                )
            else:
                # TODO
                return []
                assert False, list_item.children

        return [DefList(acc)]


def nest_sections(items):
    acc = []
    if not isinstance(items[0], Section):
        acc.append(Section([]))
    for item in items:
        if isinstance(item, Section):
            acc.append(item)
        else:
            acc[-1].children.append(item)
    for a in acc:
        a.to_json()
    return acc


def tsparse(text):
    # bytes_ = text.encode()

    tree = parser.parse(text)
    return nest_sections(TSW(text, tree.root_node).visit(tree.root_node))
