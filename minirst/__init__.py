"""
"""

__version__ = '0.0.1'

import string
import re


def parse_document(indent_lines):
    tree, rest = [], indent_lines


def parse_paragraph(lines):
    lines = list(lines)
    l0, text = lines[0]

    for ix,lx,tx in enumerate(lines[1:]):
        if lx == l0:
            text+=' '+tx
        else:
            break

    return (text, lines[ix:])







def split_indents(lines:list)-> list:
    """
    Split lines by indent level (number of leading spaces)
    """
    res = []
    for l in lines:
        trimmed = l.lstrip()
        if not trimmed:
            res.append((None,l))
        else:
            res.append((len(l)-len(trimmed),l))
    return res
    


def reformat(original):
    temp = original.split(' ')
    
    lines = []
    current_line = ''
    for word in temp:
        delta = len(word)
        current_len = len(current_line)
        if current_len+delta+1 > 80:
            lines.append(current_line)
            current_line=word
        else:
            if current_line:
                current_line +=' '
            current_line = current_line+word
    
    lines.append(current_line)
    return '\n'.join(lines)

def main():
    with open('example.rst') as f:
        original = f.read()
    print(original)
    print('-'*50)
    for l in split_indents(original.splitlines()):
        print(l)
    #print('||'+reformat(original)+'||')


if __name__ == '__main__':
    main()
