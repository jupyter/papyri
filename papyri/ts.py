import logging
from pathlib import Path
from textwrap import dedent, indent
from typing import List

from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser


allowed_adorn = "=-`:.'\"~^_*+#<>"

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
    ListItem,
    Options,
    Paragraph,
    Section,
    Strong,
    SubstitutionDef,
    SubstitutionRef,
    Transition,
    Unimplemented,
    Verbatim,
    Word,
    Words,
    compress_word,
    inline_nodes,
)

from . import errors
from .errors import (
    VisitCitationReferenceNotImplementedError,
    # VisitSubstitutionDefinitionNotImplementedError,
)

pth = str(Path(__file__).parent / "rst.so")

# replace by tree-sitter-languages once it works See https://github.com/grantjenks/py-tree-sitter-languages/issues/15
RST = Language(pth, "rst")
parser = Parser()
parser.set_language(RST)
log = logging.getLogger("papyri")


class Node:
    """
    A wrapper around tree sitter Nodes to slightly modify behavior.

    In particular we want to be able to extract whitespace information,
    which is made hard by tree sitter.

    So we intercept iterating through childrens, and if the bytes start/stop
    don't match, we insert a fake Whitespace node that has similar api to tree
    sitter official nodes.
    """

    def tree(self, bytes):
        return (
            self._repr(bytes)
            + indent("\n" + "\n".join([x.tree(bytes) for x in self.children]), "   ")
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

    def _repr(self, bytes):
        return repr(self.node) + bytes[self.start_byte : self.end_byte].decode()

    def __repr__(self):
        return f"<ts.Node {self.node.type}>"

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

    def __init__(self, bytes, root, qa):
        self.bytes = bytes
        self.root = root
        self.qa = qa
        self.depth = 0
        self._section_levels = {}
        self._targets = []

    def as_text(self, node):
        return self.bytes[node.start_byte : node.end_byte].decode()

    def visit_document(self, node):
        new_node = node.without_whitespace()
        items = self.visit(new_node)
        res = [x for x in items if not isinstance(x, Whitespace)]
        return res

    def _compressor(self, nodes):
        """
        This is currently a workaround of a tree-sitter limitations.
        List cannot have blank lines between them, so we end up with
        multiple XxxxList, instead of one XxxxList with many items.
        """

        acc = []
        current = None
        for n in nodes:
            if isinstance(n, (EnumeratedList, BulletList, FieldList, DefList)):
                if current is None:
                    current = n
                elif type(current) == type(n):
                    current.children.extend(n.children)
                else:
                    acc.append(current)
                    current = None
                    acc.append(n)

            else:
                if current:
                    acc.append(current)
                    current = None
                acc.append(n)
        if current:
            acc.append(current)

        return acc

    def _targetify(self, acc):
        """
        Here we look for targets, and if they are just
        before a section
        """
        nacc = []
        for i, a in enumerate(acc):
            if isinstance(a, Unimplemented) and a.placeholder == "untarget":
                if len(acc) > i and isinstance(acc[i + 1], Section):
                    # TODO: don't mutate. Copy
                    target_name = a.value[1:-1]
                    acc[i + 1].target = target_name
                    self._targets.append(target_name)
                    continue

            nacc.append(a)

        return nacc

    def visit(self, node):
        self.depth += 1
        acc = []
        prev_end = None
        for c in node.children:
            kind = c.type
            if kind == "::":
                if acc and isinstance(acc[-1], Word):
                    word = acc.pop()
                    acc.append(Word(word.value + ":"))
                elif acc and isinstance(acc[-1], inline_nodes):
                    acc.append(Word(":"))
                # else:
                #    assert False
                continue
            if not hasattr(self, "visit_" + kind):
                raise ValueError(
                    f"visit_{kind} not found while visiting {node}::\n{self.as_text(c)!r}"
                )
            meth = getattr(self, "visit_" + kind)
            new_children = meth(c, prev_end=prev_end)
            acc.extend(new_children)
            prev_end = c.end_point
        self.depth -= 1
        acc = self._compressor(acc)
        acc = self._targetify(acc)
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
        return [Transition()]

    def visit_reference(self, node, prev_end=None):
        """
        TODO:

        Currently we parse that as a directive, but actually it should be a reference
        and according to https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#hyperlink-references,
        there might be one or two trailing underscore, which we should pay attention to.

        """
        full_text = self.as_text(node)
        if "`" not in full_text:
            # TODO reference do not need to be in backticks,
            # though it conflict with some things like numpy
            # direct references to np.bool_, np.complex64_ (see pandas docs for example)
            # we should likely have a way to handle that.
            _text = full_text
        else:
            _text, trailing = self.as_text(node)[1:].rsplit("`", maxsplit=1)
            assert trailing in ("_", "__")
        return [Directive(_text, None, None)]

    def visit_interpreted_text(self, node, prev_end=None):
        if len(node.children) == 2:
            role, text = node.children
            assert role.type == "role"
            assert text.type == "interpreted_text"
            role_value = self.as_text(role)
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
        if role_value:
            assert ":" not in role_value
        text_value = self.as_text(text)
        assert text_value.startswith("`")
        assert text_value.endswith("`")

        inner_value = text_value[1:-1]

        if "`" in inner_value:
            log.info("issue with inner ` : %r", inner_value)
            inner_value = inner_value.replace("`", "'")

        t = Directive(
            inner_value,
            domain=domain,
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
        text = self.bytes[node.start_byte + 2 : node.end_byte - 2].decode()
        t = Verbatim([text])
        # print(' '*self.depth*4, t)
        return [t]

    def visit_literal_block(self, node, prev_end=None):
        datas = self.bytes[node.start_byte : node.end_byte].decode()
        first_offset = node.start_point[1]
        datas = " " * first_offset + datas

        b = BlockVerbatim(dedent(datas))
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
            acc.append(ListItem(self.visit(body)))
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

            pre_text = self.as_text(node.children[0])
            set_pre_a = set(pre_text)

            post_text = self.as_text(node.children[2])
            set_post_a = set(post_text)

            assert len(set_pre_a) == 1
            assert len(set_post_a) == 1

            pre_a = next(iter(set_pre_a))
            post_a = next(iter(set_post_a))

            assert pre_a == post_a

            assert len(pre_text) == len(post_text)

            assert len(pre_text) >= len(self.as_text(tc))
        else:
            assert node.children[0].type == "title"
            tc = node.children[0]
            assert node.children[1].type == "adornment"
            assert len(node.children) == 2
            pre_a = ""
            post_text = self.as_text(node.children[1])
            set_post_a = set(post_text)
            assert len(set_post_a) == 1
            post_a = next(iter(set_post_a))

            assert len(post_text) >= len(self.as_text(tc)), self.as_text(tc)

        assert post_a in allowed_adorn

        if pre_a + post_a in self._section_levels:
            level = self._section_levels[pre_a + post_a]
        else:
            level = len(self._section_levels)
            self._section_levels[pre_a + post_a] = level

        title = self.as_text(tc)
        # print(' '*self.depth*4, '== Section: ', title, '==')
        # print(' '*self.depth*4, '->', node)
        return [Section([], title, level=level)]

    def visit_block_quote(self, node, prev_end=None):
        return [BlockQuote(self.visit(node))]

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
        p = Paragraph(compress_word(acc))
        return [p, *acc2]

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
            acc.append(ListItem(self.visit(body)))
        return [EnumeratedList(acc)]

    def visit_target(self, node, prev_end=None):
        # TODO:
        # raise VisitTargetNotImplementedError()
        # self.as_text(node)
        if len(node.children) == 2:
            pp, name = node.children
            # breakpoint()
            if pp.type == ".." and name.type == "name":
                return [Unimplemented("untarget", self.as_text(name))]
        # print(node.children)
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
        text = self.bytes[node.start_byte : node.end_byte].decode()
        if "anaconda" in text:
            print("...", text)

        is_substitution_definition = False

        if len(node.children) == 4:
            kinds = [n.type for n in node.children]
            if tuple(kinds) == ("type", "::", " ", "body"):
                is_substitution_definition = True
                _role, _1, _2, body = node.children
            elif tuple(kinds) == ("..", "type", "::", "body"):
                _1, _role, _2, body = node.children
            else:
                assert False
            assert body.type == "body"
            assert _role.type == "type"
            body_children = body.children
        elif len(node.children) == 3:
            _1, _role, _2 = node.children
            body_children = []
        else:
            raise ValueError
            assert _1.type == ".."
            assert _2.type == "::"

        if _role.end_point != _2.start_point and not is_substitution_definition:
            block_data = self.bytes[node.start_byte : node.end_byte].decode()
            raise errors.SpaceAfterBlockDirectiveError(
                f"space present in {block_data!r}"
            )

        role = self.bytes[_role.start_byte : _role.end_byte].decode()
        import itertools

        groups = itertools.groupby(body_children, lambda x: x.type)
        groups = [(k, list(v)) for k, v in groups]

        if groups and groups[0][0] == "arguments":
            arg = list(groups.pop(0)[1])
            assert len(arg) == 1
            argument = self.as_text(arg[0])
        else:
            argument = ""
        if groups and groups[0][0] == "options":
            # to parse
            p0 = groups.pop(0)
            options = []
            assert len(p0[1]) == 1
            opt_node = p0[1][0]
            for field in opt_node.children:
                assert field.type == "field"
                if len(field.children) == 4:
                    c1, name, c2, body = field.children
                    options.append((self.as_text(name), self.as_text(body)))
                elif len(field.children) == 3:
                    c1, name, c2 = field.children
                    options.append((self.as_text(name), ""))
                else:
                    assert False

        else:
            options = []
        if groups and groups[0][0] == "content":
            # to parse
            content_node = list(groups.pop(0)[1])
            assert len(content_node) == 1
            content = self.as_text(content_node[0])
            padding = (content_node[0].start_point[1] - _1.start_point[1]) * " "
            content = dedent(padding + content)

        else:
            content = ""
        assert not groups
        # todo , we may want to see about the indentation of the content.

        directive = BlockDirective(role, argument, options, content)
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
        assert len(node.children) == 3
        _dotdot, sub, directive = node.children
        assert self.bytes[_dotdot.start_byte : _dotdot.end_byte].decode() == ".."
        assert sub.type == "substitution"
        assert directive.type == "directive"
        return [
            SubstitutionDef(
                self.bytes[sub.start_byte : sub.end_byte].decode(),
                self.visit_directive(directive)[0],
            )
        ]

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
                        dt=Paragraph(compress_word(self.visit(term))),
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
        acc.append(Section([], None))
    for item in items:
        if isinstance(item, Section):
            acc.append(item)
        else:
            acc[-1].children.append(item)
    return acc


def parse(text: bytes, qa=None) -> List[Section]:
    """
    Parse text using Tree sitter RST, and return a list of serialised section I guess ?
    """

    tree = parser.parse(text)
    root = Node(tree.root_node)
    tsv = TSVisitor(text, root, qa)
    res = tsv.visit_document(root)
    return nest_sections(res)


class TreeSitterParseError(Exception):
    pass
