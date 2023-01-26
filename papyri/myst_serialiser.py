"""
Unlike the current serializer (miniserde), that we use
where enums/classes are externally tagged,

Myst Json spec (https://spec.myst.tools/spec/overview) is internally tagged.

Nodes also always carry a tag for the type, even if the container/parent can
only store one type.

This is a prototype of serializer that respect this layout.
"""


from dataclasses import dataclass
from typing import List, Union
from typing import get_type_hints as gth

base_types = {int, str, bool, type(None)}


def serialize(instance, annotation):
    if annotation in base_types:
        # print("BASE", instance)
        assert isinstance(instance, annotation), f"{instance} {annotation}"
        return instance

    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        assert isinstance(instance, origin)
        inner_annotation = annotation.__args__
        # assert len(inner_annotation) == 1, inner_annotation
        return [serialize(x, inner_annotation[0]) for x in instance]
    if origin is dict:
        assert isinstance(instance, origin)
        key_annotation, value_annotation = annotation.__args__
        # assert key_annotation == str, key_annotation
        return {k: serialize(v, value_annotation) for k, v in instance.items()}
    if getattr(annotation, "__origin__", None) is Union:
        inner_annotation = annotation.__args__
        if len(inner_annotation) == 2 and inner_annotation[1] == type(None):
            # assert inner_annotation[0] is not None
            # here we are optional; we _likely_ can avoid doing the union trick and store just the type, or null
            if instance is None:
                return None
            else:
                return serialize(instance, inner_annotation[0])
        assert (
            type(instance) in inner_annotation
        ), f"{type(instance)} not in {inner_annotation}, {instance} or type {type(instance)}"
        ma = [x for x in inner_annotation if type(instance) is x]
        # assert len(ma) == 1
        ann_ = ma[0]
        return {"type": ann_.__name__, "data": serialize(instance, ann_)}
    if (
        (type(annotation) is type)
        and type.__module__ not in ("builtins", "typing")
        and (instance.__class__.__name__ == getattr(annotation, "__name__", None))
        or type(instance) == annotation
    ):
        data = {}
        data["type"] = type(instance).__name__
        for k, ann in gth(type(instance)).items():
            data[k] = serialize(getattr(instance, k), ann)
        return data
    # print(
    #    instance,
    #    (type(annotation) is type),
    #    type.__module__,
    #    type(instance) == annotation,
    # )


if __name__ == "__main__":

    @dataclass
    class Bar:
        x: str
        y: Union[int, bool]

    @dataclass
    class Foo:
        a: int
        b: List[int]
        c: Bar

    f = Foo(1, [1, 3], Bar("str", False))
    import json

    print(json.dumps(serialize(f, Foo), indent=2))
