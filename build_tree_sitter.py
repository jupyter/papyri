from tree_sitter import Language, Parser

Language.build_library(
    # Store the library in the `build` directory
    "papyri/rst.so",
    # Include one or more languages
    [
        "tree-sitter-rst",
    ],
)

PY_LANGUAGE = Language("papyri/rst.so", "rst")
