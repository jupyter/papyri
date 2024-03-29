"""
Various directive handlers.
"""

from .myst_ast import MMath, MAdmonition, MAdmonitionTitle, MText
from .ts import parse
from .take2 import Section
import logging

log = logging.getLogger("papyri")


def block_math_handler(argument, options, content):
    """
    Handler for the block math directive handler.
    """
    if argument and content:
        log.info(
            "For consistency, please use the math directive"
            " with all the equations in the content of the directive",
        )
        content = argument + content
    elif argument and not content:
        # TODO: do we want to allow that ?
        content = argument
    return [MMath(content)]


#  A number of directives that so far are just small wrappers around admonitions.


def admonition_helper(name, argument, options, content):
    """
    This is a helper to return admonition.
    """
    assert not options
    if content:
        inner = parse(content.encode(), qa="")
        assert len(inner) == 1

        assert isinstance(inner[0], Section)

        return [
            MAdmonition(
                [MAdmonitionTitle([MText(f"{name} {argument}")])] + inner[0].children,
                kind=name,
            )
        ]
    else:
        return [
            MAdmonition([MAdmonitionTitle([MText(f"{name} {argument}")])], kind=name)
        ]


def warning_handler(argument, options, content):
    return admonition_helper("warning", argument, options, content)


def note_handler(argument, options, content):
    return admonition_helper("note", argument, options, content)


def versionadded_handler(argument, options, content):
    return admonition_helper("versionadded", argument, options, content)


def versionchanged_handler(argument, options, content):
    return admonition_helper("versionchanged", argument, options, content)


def deprecated_handler(argument, options, content):
    return admonition_helper("deprecated", argument, options, content)
