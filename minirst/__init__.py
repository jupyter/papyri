"""
"""

import sys
from pathlib import Path

__version__ = '0.0.1'

def parse(input):
    """
    parse an input string into token/tree.

    FOr now only return a list of tokens

    """
    tokens=[]
    for l in input.splitlines():
        tokens.extend(l.split(' '))
    return tokens


def transform(tokens):
    """
    Accumulate tokens in lines.

    Add token (and white spaces) to a line until it overflow 80 chars.
    """
    lines = []
    current_line = []
    for t in tokens:
        if sum([len(x)+1 for x in current_line])+len(t) > 80:
            lines.append(current_line)
            current_line=[]
        current_line.append(t)
    if current_line:
        lines.append(current_line)
    return lines


def format(lines):
    s = ''
    for l in lines:
        s = s+ ' '.join(l)
        s = s+'\n'
    return s[:-1]




def reformat(input):
    return format(transform(parse(format(transform(parse(input))))))

def main():
    filename = sys.argv[1]
    print(f'processing {filename}')
    p = Path(filename)

    with p.open() as f:
        data = f.read()
        print(reformat)
    


if __name__ == '__main__':
    main()
