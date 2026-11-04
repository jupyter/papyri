"""
This file is meant to test the serialisation / deserialization
of function signature to JSON.
"""

import json
from collections.abc import Callable
from typing import Any

import pytest

from papyri.signature import Signature as SignatureObject
from papyri.signature import SignatureNode

all_funcs: list[Callable[..., Any]] = []


def add(func: Callable[..., Any]) -> Callable[..., Any]:
    all_funcs.append(func)
    return func


@add
def function_1(posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs):
    """{
      "kind": "function",
      "parameters": [
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_ONLY",
          "name": "posonly",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_or_k",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "data": "1",
            "type": "str"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_ok_k_d",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_POSITIONAL",
          "name": "varargs",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_KEYWORD",
          "name": "varkwargs",
          "type": "SigParam"
        }
      ],
      "return_annotation": {"type": "Empty"},
      "target_name": "function_1",
      "type": "signature"
    }"""
    pass


@add
def async_function_2(posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs):
    """{
      "kind": "function",
      "parameters": [
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_ONLY",
          "name": "posonly",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_or_k",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "data": "1",
            "type": "str"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_ok_k_d",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_POSITIONAL",
          "name": "varargs",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_KEYWORD",
          "name": "varkwargs",
          "type": "SigParam"
        }
      ],
      "return_annotation": {"type": "Empty"},
      "target_name": "async_function_2",
      "type": "signature"
    }"""
    pass


@add
def generator_function_3(posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs):
    """{
      "kind": "generator function",
      "parameters": [
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_ONLY",
          "name": "posonly",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_or_k",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "data": "1",
            "type": "str"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_ok_k_d",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_POSITIONAL",
          "name": "varargs",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_KEYWORD",
          "name": "varkwargs",
          "type": "SigParam"
        }
      ],
      "return_annotation": {"type": "Empty"},
      "target_name": "generator_function_3",
      "type": "signature"
    }"""
    yield


@add
async def async_generator_function_4(
    posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs
):
    """{
      "kind": "async_generator function",
      "parameters": [
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_ONLY",
          "name": "posonly",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_or_k",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "data": "1",
            "type": "str"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_ok_k_d",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_POSITIONAL",
          "name": "varargs",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_KEYWORD",
          "name": "varkwargs",
          "type": "SigParam"
        }
      ],
      "return_annotation": {"type": "Empty"},
      "target_name": "async_generator_function_4",
      "type": "signature"
    }"""
    yield


@add
async def coroutine_function_5(
    posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs
):
    """{
      "kind": "coroutine function",
      "parameters": [
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_ONLY",
          "name": "posonly",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_or_k",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "data": "1",
            "type": "str"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_ok_k_d",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_POSITIONAL",
          "name": "varargs",
          "type": "SigParam"
        },
        {
          "annotation": {
            "type": "Empty"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "VAR_KEYWORD",
          "name": "varkwargs",
          "type": "SigParam"
        }
      ],
      "return_annotation": {"type": "Empty"},
      "target_name": "coroutine_function_5",
      "type": "signature"
    }"""
    pass


@add
def function_with_annotation5(a: int, b: int | float) -> bool | None:
    """
    {
      "kind": "function",
      "parameters": [
        {
          "annotation": {
            "data": "int",
            "type": "str"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "a",
          "type": "SigParam"
        },
        {
          "annotation": {
            "data": "int | float",
            "type": "str"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "b",
          "type": "SigParam"
        }
      ],
      "return_annotation": {
          "data": "bool | None",
          "type": "str"
      },
      "target_name":"function_with_annotation5",
      "type": "signature"
    }

    """
    pass


@pytest.mark.parametrize(
    "func",
    all_funcs,
)
def test_f1(func: Callable[..., Any]) -> None:
    so = SignatureObject(func)
    node = so.to_node()
    bytes_ = node.to_json()
    assert json.dumps(json.loads(bytes_), indent=2) == json.dumps(
        json.loads(func.__doc__ or ""), indent=2
    )
    node_back = SignatureNode.from_json(bytes_)
    assert node_back == node


def test_string_defaults_keep_quotes() -> None:
    def func(sep=",", fill=" ", newline="\n", empty="", count=1, none=None):
        pass

    defaults = {p.name: p.default for p in SignatureObject(func).to_node().parameters}
    assert defaults == {
        "sep": "','",
        "fill": "' '",
        "newline": "'\\n'",
        "empty": "''",
        "count": "1",
        "none": "None",
    }
