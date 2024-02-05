from __future__ import annotations

import json
import typing
from typing import Dict, Any

import cbor2

from .miniserde import get_type_hints, deserialize
from .myst_serialiser import serialize as myst_serialize


class Base:
    def validate(self):
        validate(self)
        return self

    @classmethod
    def _instance(cls):
        return cls()


class Node(Base):
    def __init__(self, *args, **kwargs):
        tt = get_type_hints(type(self))
        if type(self).__name__ == "MMystDirective":
            tt = {k: v for k, v in tt.items() if k != "type"}
        for attr, val in zip(tt, args):
            setattr(self, attr, val)
        for k, v in kwargs.items():
            assert k in tt, f"{k} not in {tt}"
            setattr(self, k, v)
        if hasattr(self, "_post_deserialise"):
            self._post_deserialise()

    def cbor(self, encoder):
        tag = TAG_MAP[type(self)]
        attrs = get_type_hints(type(self))
        encoder.encode(cbor2.CBORTag(tag, [getattr(self, k) for k in attrs]))

    def __eq__(self, other):
        if not (type(self) == type(other)):
            return False
        tt = get_type_hints(type(self))
        for attr in tt:
            a, b = getattr(self, attr), getattr(other, attr)
            if a != b:
                return False

        return True

    def __repr__(self):
        tt = get_type_hints(type(self))
        acc = ""
        for t in tt:
            acc += f"{t}: {getattr(self, t)!r}\n"

        return f"<{self.__class__.__name__}: \n{indent(acc)}>"

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True).encode()

    @classmethod
    def from_json(cls, data: bytes):
        return cls.from_dict(json.loads(data))

    def to_dict(self):
        return myst_serialize(self, type(self))

    @classmethod
    def from_dict(cls, data):
        return deserialize(cls, cls, data)

    def __hash__(self):
        return hash(
            tuple(
                tuple(getattr(self, x))
                for x in dir(self)
                if not x.startswith("_") and not callable(getattr(self, x))
            )
        )


class UnserializableNode(Node):
    _dont_serialise = True

    def cbor(self, encoder):
        assert False

    def to_json(self) -> bytes:
        assert False


TAG_MAP: Dict[Any, int] = {}
REV_TAG_MAP: Dict[int, Any] = {}


def indent(text, marker="   |"):
    """
    Return the given text indented with 3 space plus a pipe for display.
    """
    lines = text.split("\n")
    return "\n".join(marker + l for l in lines)


def _invalidate(obj, depth=0):
    """
    Recursively validate type anotated classes.
    """

    annotations = get_type_hints(type(obj))
    for k, v in annotations.items():
        # FIX: AttributeError: 'MText' object has no attribute 'position'
        item = getattr(obj, k)
        res = not_type_check(item, v)
        if res:
            return f"{k} field of  {type(obj)} : {res}"

        if isinstance(item, (list, tuple)):
            for ii, i in enumerate(item):
                sub = _invalidate(i, depth + 1)
                if sub is not None:
                    return f"{k}.{ii}." + sub
        if isinstance(item, dict):
            for ii, i in item.items():
                sub = _invalidate(i, depth + 1)
                if sub is not None:
                    return f"{k}.{ii}." + sub
        else:
            sub = _invalidate(item, depth + 1)
            if sub is not None:
                return f"{k}.{sub}." + sub

    # return outcome,s


class WrongTypeAtField(ValueError):
    pass


def validate(obj: Any) -> None:
    res = _invalidate(obj)
    if res:
        raise WrongTypeAtField(f"Wrong type at field :: {res}")


def not_type_check(item, annotation):
    if not hasattr(annotation, "__origin__"):
        if isinstance(item, annotation):
            return None
        else:
            return f"expecting {annotation} got {type(item)} : {item!r}"
    elif annotation.__origin__ is dict:
        if not isinstance(item, dict):
            return f"got  {type(item)}, Yexpecting list"
        inner_type = annotation.__args__[0]
        a = [not_type_check(i, inner_type) for i in item.keys()]
        ax = [x for x in a if x is not None]
        inner_type = annotation.__args__[1]
        b = [not_type_check(i, inner_type) for i in item.values()]
        bx = [x for x in b if x is not None]
        if ax:
            return ":invalid key type {ax[0]}"
        if bx:
            return bx[0]
        return None
    elif annotation.__origin__ in (list, tuple):
        # technically incorrect
        if not isinstance(item, (list, tuple)):
            return f"got  {type(item)}, Yexpecting list"
        # todo, this does not support Tuple[x,x] < len of tuple, and treat is as a list.
        assert len(annotation.__args__) == 1
        inner_type = annotation.__args__[0]

        b = [not_type_check(i, inner_type) for i in item]

        bp = [x for x in b if x is not None]
        if bp:
            return bp[0]
        else:
            return None
    elif annotation.__origin__ is typing.Union:
        if any([not_type_check(item, arg) is None for arg in annotation.__args__]):
            return None
        return f"expecting one of {annotation!r}, got {item!r}"
    raise ValueError(item, annotation)


def register(value):
    assert value not in REV_TAG_MAP, REV_TAG_MAP[value]

    def _inner(type_):
        assert type_ not in TAG_MAP
        TAG_MAP[type_] = value
        REV_TAG_MAP[value] = type_

        return type_

    return _inner
