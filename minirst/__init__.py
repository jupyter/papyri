"""
"""

__version__ = '0.0.1'

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
    print('||'+reformat(original)+'||')


if __name__ == '__main__':
    main()
