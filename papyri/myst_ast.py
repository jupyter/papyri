"""
An attempt to create AST from MyST spec. At the moment just trying to
add Text from Myst to the generated structure in the ts.py.

"""
from papyri.common_ast import Node


# Prefixed with "M" as Text exists in take2.py as well
class MText(Node):
    type = 'text'
    value: str
    # position: Any
    # data: Any
