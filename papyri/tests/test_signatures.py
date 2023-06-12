"""
This file is meant to test the serialisation / deserialisation 
of function signture to JSON.
"""


def function_1(posonly, /, pos_or_k, pos_ok_k_d=1, *varargs, **varkwargs):
    pass


def test_f1():
    from papyri.signature import Signature as SignatureObject

    so = SignatureObject(function_1)
    assert so.to_node().to_json() == function_1.__doc__
