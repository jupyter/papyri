import logging
import itertools
from textwrap import dedent, indent
from typing import List, Any, Dict


from .myst_ast import (
    MText,
    MCode,
    MParagraph,
    MEmphasis,
    MInlineCode,
    MStrong,
    MList,
    MListItem,
    UnprocessedDirective,
    MComment,
    MBlockquote,
)


from .take2 import (
    DefList,
    DefListItem,
    Directive,
    FieldList,
    FieldListItem,
    Options,
    Section,
    SubstitutionDef,
    SubstitutionRef,
    MThematicBreak,
    Unimplemented,
    compress_word,
    inline_nodes,
)

from . import errors
from .errors import (
    VisitCitationReferenceNotImplementedError,
    # VisitSubstitutionDefinitionNotImplementedError,
)


from tree_sitter_languages import get_parser

parser = get_parser("rst")
allowed_adorn = "=-`:.'\"~^_*+#<>"

log = logging.getLogger("papyri")


class Node:
    """
    A wrapper around tree sitter Nodes to slightly modify behavior.

    In particular we want to be able to extract whitespace information,
    which is made hard by tree sitter.

    So we intercept iterating through children, and if the bytes start/stop
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

        current_byte = self.start_byte
        current_point = self.start_point

        new_nodes = []
        if self.node.children:
            for n in self.node.children:
                if n.start_byte != current_byte:
                    # value = self._bytes[current_byte:n.start_byte]
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
        return self.node._bytes

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
    Tree sitter Visitor

    Walk the tree sitter tree and convert each node into our kind of internal node.

    """

    _bytes: bytes
    _qa: str
    _section_levels: Dict[str, int]
    _targets: List[str]

    def __init__(self, buf: bytes, qa: str, /):
        """
        A tree-visitor for TreeSitter nodes to convert into Papyri/Myst nodes.

        Parameters
        ----------
        buf: bytes
            bytes buffer of the document  parsed by tree-sitter.
        qa: str
            fully qualifed name of the object for whcih we are paring the
            documentation. This is used only in log messages, and for debugging purpose.
        """
        self._bytes = buf
        assert qa is not None
        self._qa = qa
        self._section_levels = {}
        self._targets = []

    def as_text(self, node) -> str:
        """
        Utility function to extract the original text for a given node.
        """
        return self._bytes[node.start_byte : node.end_byte].decode()

    def visit_document(self, node):
        new_node = node.without_whitespace()
        items = self.visit(new_node)
        res = [x for x in items if not isinstance(x, Whitespace)]
        return res

    def _compressor(self, nodes) -> List[Any]:
        """
        This is currently a workaround of a tree-sitter limitations.
        List cannot have blank lines between them, so we end up with
        multiple XxxxList, instead of one XxxxList with many items.
        """

        acc = []
        current = None
        for n in nodes:
            if isinstance(n, (FieldList, DefList, MList)):
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
        acc = []
        # TODO: FIX
        if node.type == "ERROR":
            print(f"ERROR node: {self.as_text(node)!r}, skipping")
            return []
        for c in node.children:
            # c=<ts.Node directive>
            kind = c.type
            if kind == "::":
                if acc and isinstance(acc[-1], inline_nodes):
                    acc.append(MText(":"))
                # else:
                #    assert False
                continue
            if not hasattr(self, "visit_" + kind):
                raise ValueError(
                    f"visit_{kind} not found while visiting {node}::\n{self.as_text(c)!r}"
                )
            meth = getattr(self, "visit_" + kind)
            new_children = meth(c)
            acc.extend(new_children)
        acc = self._compressor(acc)
        acc = self._targetify(acc)
        return acc

    def visit_citation(self, node):
        # raise VisitCitationNotImplementedError()
        # just hlines, like ------
        return [Unimplemented("citation", self.as_text(node))]

    def visit_citation_reference(self, node):
        raise VisitCitationReferenceNotImplementedError()
        # just hlines, like ------
        return []

    def visit_transition(self, node):
        return [MThematicBreak()]

    def visit_reference(self, node):
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
            _text, trailing = (
                self.as_text(node)[1:].replace("\n", " ").rsplit("`", maxsplit=1)
            )
            assert trailing in ("_", "__")
        return [Directive(_text, None, None)]

    def visit_interpreted_text(self, node):
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
                # TODO: error for pandas.io.orc:read_orc
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

        inner_value = text_value[1:-1].replace("\n", " ")

        if "`" in inner_value:
            log.info(
                "Improper backtick found in interpreted text. "
                "This is usually due to a missing/stray backtick, or "
                "missing escape (`\\`) on trailing charter : %r in (%s)",
                inner_value,
                self._qa,
            )
            log.warning("replacing ` by ' to not crash serialiser")
            inner_value = inner_value.replace("`", "'")

        t = Directive(
            inner_value,
            domain=domain,
            role=role_value,
        )
        return [t]

    def visit_standalone_hyperlink(self, node):
        return self.visit_text(node)

    def visit_text(self, node):
        text = self.as_text(node)
        assert not text.startswith(":func:"), breakpoint()
        t = MText(text)
        return [t]

    def visit_whitespace(self, node):
        """
        Here whitespace seem to mean both spaces and newline.

        I believe we need to remove the newline, but there is some questions as
        to whether we should replace them with just space, or suppress them if
        preceded by spaced.
        """
        content = self.as_text(node)
        # assert set(content) in ({" "}, {"\n"}), repr(content)
        t = MText(" " * len(content))
        return [t]

    def visit_literal(self, node):
        text = self.as_text(node)[2:-2]
        assert "\n\n" not in text
        t = MInlineCode(text.replace("\n", " "))
        return [t]

    def visit_literal_block(self, node):
        datas = self.as_text(node)
        first_offset = node.start_point[1]
        datas = " " * first_offset + datas
        b = MCode(dedent(datas))
        return [b]

    def visit_bullet_list(self, node):
        myst_acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
            assert len(list_item.children) == 2, list_item.children
            _bullet, body = list_item.children
            # assert len(body.children) == 1
            # parg = body.children[0]
            # assert parg.type == "paragraph", parg.type
            myst_acc.append(MListItem(False, self.visit(body)))
        return [MList(ordered=False, start=1, spread=False, children=myst_acc)]

    def visit_section(self, node):
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
            # this triggers sometime because tree sitter missparse a few things like.
            #
            # ```
            # blah blah reference and new line with parenthesis
            # (like a year here)
            # ```
            assert len(set_post_a) == 1, breakpoint()
            post_a = next(iter(set_post_a))
            # TODO: fails with pandas.compat._constants
            assert len(post_text) >= len(self.as_text(tc)), self.as_text(tc)

        assert post_a in allowed_adorn

        if pre_a + post_a in self._section_levels:
            level = self._section_levels[pre_a + post_a]
        else:
            level = len(self._section_levels)
            self._section_levels[pre_a + post_a] = level

        title = self.as_text(tc)
        return [Section([], title, level=level)]

    def visit_block_quote(self, node):
        return [MBlockquote(self.visit(node))]

    def visit_paragraph(self, node):
        sub = self.visit(node.with_whitespace())
        acc = []
        acc2 = []

        for item in sub:
            if isinstance(item, MCode):
                acc2.append(item)
                continue
            acc.append(item)
        if acc[-1] == MText(" "):
            acc.pop()
        assert len(acc2) < 2
        # p = Paragraph(compress_word(acc))
        p = MParagraph(compress_word(acc))
        return [p, *acc2]

    def visit_line_block(self, node):
        # TODO
        # e.g: numpy/doc/source/user/c-info.how-to-extend.rst
        log.warning("Skipping unimplemented line_block node: %s", self.as_text(node))
        return []

    def visit_substitution_reference(self, node):
        # TODO
        return [SubstitutionRef(self.as_text(node))]

    def visit_doctest_block(self, node) -> List[MCode]:
        # TODO
        return self.visit_literal_block(node)

    def visit_field(self, node):
        return []

    def visit_field_list(self, node) -> List[FieldList]:
        acc: List[str] = []

        lens = {len(f.children) for f in node.children}
        if lens == {3}:  # need test here don't know why it was here.
            # we likely have an option list
            for list_item in node.children:
                assert list_item.type == "field"
                col1, name, col2 = list_item.children
                # TODO, assert _ and _ are `:`
                assert self.as_text(col1) == ":", col1
                assert self.as_text(col2) == ":", col2
                acc.append(self.as_text(name))
            return []
            # TODO: why do we have unreachable here
            return [Options(acc)]
        acc2: List[FieldListItem] = []
        if lens == {4}:
            for list_item in node.children:
                assert list_item.type == "field"
                _, name, _, body = list_item.children
                a, b = compress_word(self.visit(name)), compress_word(self.visit(body))
                # [_.to_json()for _ in a]
                # [_.to_json() for _ in b]
                f = FieldListItem(a, b)
                acc2.append(f)
            return [FieldList(acc2)]

        raise ValueError("mixed len...")

    def visit_enumerated_list(self, node):
        myst_acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
            _bullet, body = list_item.children
            myst_acc.append(MListItem(False, self.visit(body)))
        return [MList(ordered=True, start=1, spread=False, children=myst_acc)]

    def visit_target(self, node):
        # TODO:
        # raise VisitTargetNotImplementedError()
        # self.as_text(node)
        if len(node.children) == 2:
            pp, name = node.children
            # breakpoint()
            if pp.type == ".." and name.type == "name":
                return [Unimplemented("untarget", self.as_text(name))]
        return [Unimplemented("target", self.as_text(node))]

    # def visit_arguments(self, node):
    #    assert False
    #    return []

    def visit_attribution(self, node):
        # TODO:
        print("attribution not implemented")
        return [Unimplemented("inline_target", self.as_text(node))]

    def visit_inline_target(self, node):
        # TODO:
        print("inline_target not implemented")
        return [Unimplemented("inline_target", self.as_text(node))]

    def visit_directive(self, node):
        """
        Main entry point for directives.

        Parses directive arguments, options and content into a UnprocessedDirective
        object.

        Parameters
        ----------
        node: Node
            The directive to parse

        Returns
        -------
        directive: UnprocessedDirective


        Notes
        -----

        The way tree sitter works when parsing directive, we will get a sequence
        of nodes of various types, thus we must iterate over them and look at
        their types, which is efficient in term of spaces, but as convenient
        as term of API, thus will try to disambiguate the various case we
        encounter to form something a bit more coherent. We want to have structs
        which all have the same fields, and the fields potentially emtpy if
        absent.


        Note though that some of the directive do allow to now have any
        arguments, or option and allow the content to start just after the
        ``::`` marker, and we will need some spacial casing/ workaround for
        this.

        """
        # TODO:
        # maybe make it part of the type if a block directive (has, or not), a body

        is_substitution_definition = False

        if len(node.children) == 4:
            # This directive has a body
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
            raw = self.as_text(body)
        elif len(node.children) == 3:
            # this rarely happens, but for example, we can find ``.. toctree::`` empty directive.
            _1, _role, _2 = node.children
            body_children = []
            raw = ""
        else:
            raise ValueError(f"Wrong number of children: {len(node.children)}")

        # Sphinx / docutils a bit more lenient
        if _role.end_point != _2.start_point and not is_substitution_definition:
            raise errors.SpaceAfterBlockDirectiveError(
                f"space present in {self.as_text(node)!r}"
            )

        role = self.as_text(_role)

        groups = itertools.groupby(body_children, lambda x: x.type)
        groups = [(k, list(v)) for k, v in groups]
        for k, _ in groups:
            assert k in {"arguments", "content", "options"}, k

        if role == "warning":
            # The warning directive does not take a title argument;
            # however, the contents for the directive may be defined inline
            # with the directive name, or as a separate block.
            # See https://docutils.sourceforge.io/docs/ref/doctree.html#warning
            if len(groups) == 1:
                content_node = list(groups[0][1])
                content = self.as_text(content_node[0])
            elif len(groups) == 2:
                content_node = [groups[0][1][0], groups[1][1][0]]
                content = (
                    self.as_text(content_node[0]) + " " + self.as_text(content_node[1])
                )
            else:
                raise ValueError(f"{role} directive has no content")

            padding = (content_node[0].start_point[1] - _1.start_point[1]) * " "
            content = dedent(padding + content).lstrip(" ")
            argument = ""
            options = []
            groups = []
            children = []

        else:
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
                        _c1, name, _c2, body = field.children
                        options.append((self.as_text(name), self.as_text(body)))
                    elif len(field.children) == 3:
                        _c1, name, _c2 = field.children
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
                children = self.visit(content_node[0])
            else:
                content = ""
                children = []

        assert not groups
        # todo , we may want to see about the indentation of the content.

        directive = UnprocessedDirective(
            role, argument, dict(options), content, children=children, raw=raw
        )
        return [directive]

    def visit_footnote_reference(self, node):
        # TODO
        # assert False, self.as_text(node)
        return []

    def visit_emphasis(self, node):
        # TODO
        return [MEmphasis([MText(self.as_text(node)[1:-1])])]

    def visit_substitution_definition(self, node):
        assert len(node.children) == 3
        _dotdot, sub, directive = node.children
        assert self.as_text(_dotdot) == ".."
        assert sub.type == "substitution"
        assert directive.type == "directive"
        return [
            SubstitutionDef(
                value=self.as_text(sub),
                children=self.visit_directive(directive),
            )
        ]

    def visit_comment(self, node):
        # TODO
        return [MComment(self.as_text(node))]
        # raise VisitCommentNotImplementedError()

    def visit_strong(self, node):
        return [MStrong([MText(self.as_text(node)[2:-2])])]

    def visit_footnote(self, node):
        # TODO
        # that is actually used for references
        # assert False, self.as_text(node)
        return [Unimplemented("footnote", self.as_text(node))]

    def visit_ERROR(self, node):
        """
        Called with parsing error nodes.
        """
        # TODO
        # raise TreeSitterParseError()
        return []

    def visit_definition_list(self, node):
        acc = []
        for list_item in node.children:
            assert list_item.type == "list_item"
            if len(list_item.children) == 2:
                term, definition = list_item.children
                assert term.type == "term"
                assert definition.type == "definition"
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
                        dt=MParagraph(compress_word(self.visit(term))),
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
    res = TSVisitor(text, qa).visit_document(root)
    ns = nest_sections(res)
    return ns


class TreeSitterParseError(Exception):
    pass
