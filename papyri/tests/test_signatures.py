"""
This file is meant to test the serialisation / deserialization
of function signature to JSON.
"""
from papyri.signature import Signature as SignatureObject, SignatureNode
import json
from typing import Union, Optional

import pytest

all_funcs = []


def add(func):
    global all_funcs
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
        }
      ],
      "return_annotation": {"type": "Empty"},
      "type": "SignatureNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
        }
      ],
      "return_annotation": {"type": "Empty"},
      "type": "SignatureNode"
    }"""
    pass


@add
def generator_function_3(posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs):
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
        }
      ],
      "return_annotation": {"type": "Empty"},
      "type": "SignatureNode"
    }"""
    yield


@add
async def async_generator_function_4(
    posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs
):
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
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
          "type": "ParameterNode"
        }
      ],
      "return_annotation": {"type": "Empty"},
      "type": "SignatureNode"
    }"""
    yield


@add
def function_with_annotation5(a: int, b: Union[int, float]) -> Optional[bool]:
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
          "type": "ParameterNode"
        },
        {
          "annotation": {
            "data": "Union[int, float]",
            "type": "str"
          },
          "default": {
            "type": "Empty"
          },
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "b",
          "type": "ParameterNode"
        }
      ],
      "return_annotation": {
          "data": "typing.Optional[bool]",
          "type": "str"
      },
      "type": "SignatureNode"
    }

    """
    pass


@pytest.mark.parametrize(
    "func",
    all_funcs,
)
def test_f1(func):
    so = SignatureObject(func)
    node = so.to_node()
    bytes_ = node.to_json()
    assert json.dumps(json.loads(bytes_), indent=2) == json.dumps(
        json.loads(func.__doc__), indent=2
    )
    node_back = SignatureNode.from_json(bytes_)
    assert node_back == node
