from collections import namedtuple
from itertools import chain, cycle

import black


def reformat(lines, indent=4):
    text = "\n".join(lines)
    if "doctest:" in text:
        return lines
    try:
        mode = black.FileMode()
        mode.line_length -= indent + 4
        return black.format_str(text, mode=black.FileMode()).splitlines()
    except Exception as e:
        raise ValueError("could not reformat:" + repr(text)) from e


def insert_promt(lines):
    new = []
    for p, l in zip(chain([">>> "], cycle(["... "])), lines):
        new.append(p + l)
    return new


def splitblank(list):
    items = []
    current = []
    for l in list:
        if not l.strip():
            if current:
                items.append(current)
            current = []
        else:
            current.append(l)
    if current:
        items.append(current)
    return items


InOut = namedtuple("InOut", ["in_", "out"])
Text = namedtuple("Text", ["in_", "out"])


def InOutText(a, b):
    if not a:
        return Text(a, b)
    else:
        return InOut(a, b)


def splitcode(lines):
    """
    Split a block of lines without blank lines into categories.

    Code lines start with >>> or ...,
    then outputs, start with none of the two above.

    """
    items = []
    in_ = []
    out = []
    if not lines[0].startswith(">>>"):
        return [InOutText([], lines)]

    state = "code"
    for i, l in enumerate(lines):
        if l.startswith(">>> "):
            state = "code"
            if in_ or out:
                items.append(InOutText(in_, out))
            in_, out = [], []

            in_.append(l[4:])
        # ... can appear in pandas output.
        elif l.startswith("... ") and state == "code":
            in_.append(l[4:])
        else:
            state = "notcode"
            out.append(l)
    if in_ or out:
        items.append(InOutText(in_, out))
    return items


def reformat_example_lines(ex, indent=4):
    from there import print

    oo = []
    # print(ex)
    try:
        blocks = splitblank(ex)
        for block in blocks:
            # print(block)
            codes = splitcode(block)
            for (in_, out) in codes:
                oo.extend(insert_promt(reformat(in_, indent=4)))
                if out:
                    oo.extend(out)
            oo.append("")
        return oo[:-1]
    except Exception:
        print(block)
        import sys

        raise
