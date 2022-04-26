def flatten(dct):
    return {k: [s for sub in toc for s in sub] for k, toc in dct.items()}


def dotdotcount(path):
    n = 0
    acc = []
    leading = True
    for it in path:
        if it == "..":
            assert leading is True, path
            n += 1
        else:
            leading = False
            acc.append(it)
    return n, acc


def _tree(current_path, unnest, counter, depth=0):
    counter[current_path] += 1
    children = {}
    children_path = unnest.get(current_path, [])
    directory = current_path.split(":")[:-1]
    # print(' '*depth*4, 'dir', directory, f'({current_path})')
    for cp in children_path:

        assert not cp.startswith("/")
        if cp.endswith("/"):
            cp = cp + "index"

        if cp.startswith("https://"):
            continue

        n, sub = dotdotcount(cp.split("/"))
        directory = current_path.split(":")[: -1 - n]
        p = ":".join(directory + sub)

        if p.endswith(".rst"):
            p = p[:-4]
        assert p != current_path
        # print(' '*depth*4,cp, '->', p)
        assert p not in children
        children[p] = _tree(p, unnest, depth=depth + 1, counter=counter)

    return children


def make_tree(data):
    data = {k: v for k, v in data.items()}
    data = flatten(data)
    data = {k: [i[1] for i in v] for k, v in data.items()}
    c = {k: 0 for k in data.keys()}
    my_tree = _tree("index", data, c)
    return my_tree
