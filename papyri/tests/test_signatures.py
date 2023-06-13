"""
This file is meant to test the serialisation / deserialisation 
of function signture to JSON.
"""
from papyri.signature import Signature as SignatureObject, SignatureNode
import json

import pytest

all_funs = []


def add(func):
    global all_funs
    all_funs.append(func)
    return func


@add
def function_1(posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs):
    """
    {
      "kind": "function",
      "parameters": [
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_ONLY",
          "name": "posonly",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_or_k",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_ok_k_d",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "VAR_POSITIONAL",
          "name": "varargs",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "VAR_KEYWORD",
          "name": "varkwargs",
          "type": "ParameterNode"
        }
      ],
      "type": "SignatureNode"
    }"""
    pass


@add
def async_function_1(posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs):
    """{
      "kind": "function",
      "parameters": [
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_ONLY",
          "name": "posonly",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_or_k",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_ok_k_d",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "VAR_POSITIONAL",
          "name": "varargs",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "VAR_KEYWORD",
          "name": "varkwargs",
          "type": "ParameterNode"
        }
      ],
      "type": "SignatureNode"
    }"""
    pass


@add
def generator_function_1(posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs):
    """{
      "kind": "function",
      "parameters": [
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_ONLY",
          "name": "posonly",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_or_k",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_ok_k_d",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "VAR_POSITIONAL",
          "name": "varargs",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "VAR_KEYWORD",
          "name": "varkwargs",
          "type": "ParameterNode"
        }
      ],
      "type": "SignatureNode"
    }"""
    yield


@add
async def async_generator_function_1(
    posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs
):
    """{
      "kind": "function",
      "parameters": [
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_ONLY",
          "name": "posonly",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_or_k",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "POSITIONAL_OR_KEYWORD",
          "name": "pos_ok_k_d",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "VAR_POSITIONAL",
          "name": "varargs",
          "type": "ParameterNode"
        },
        {
          "annotation": null,
          "default": null,
          "kind": "VAR_KEYWORD",
          "name": "varkwargs",
          "type": "ParameterNode"
        }
      ],
      "type": "SignatureNode"
    }"""
    yield


@pytest.mark.parametrize(
    "func",
    all_funs,
)
def test_f1(func):
    so = SignatureObject(func)
    node = so.to_node()
    bytes_ = node.to_json()
    assert json.loads(bytes_) == json.loads(func.__doc__)
    node_back = SignatureNode.from_json(bytes_)
    assert node_back == node
