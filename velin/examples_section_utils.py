import black


def reformat(lines):
    text = "\n".join(lines)
    return black.format_str(text, mode=black.FileMode()).splitlines()


from itertools import cycle, chain


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


def splitcode(lines):
    items = []
    in_ = []
    out = []
    if not lines[0].startswith(">>>"):
        return [([], lines)]
    for i, l in enumerate(lines):
        if l.startswith(">>> "):
            if in_ or out:
                items.append((in_, out))
            in_, out = [], []

            in_.append(l[4:])
        elif l.startswith("..."):
            in_.append(l[4:])
        else:
            out.append(l)
    if in_ or out:
        items.append((in_, out))
    return items


def reformat_example_lines(ex):
    oo = []
    blocks = splitblank(ex)
    for block in blocks:
        codes = splitcode(block)
        for (in_, out) in codes:
            # print(in_, out)
            oo.extend(insert_promt(reformat(in_)))
            if out:
                oo.extend(out)
        oo.extend(" ")
    return oo
