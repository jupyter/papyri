"""
Unlike the current serializer (miniserde), that we use
where enums/classes are externally tagged,

Myst Json spec (https://spec.myst.tools/spec/overview) is internally tagged.

Nodes also always carry a tag for the type, even if the container/parent can
only store one type.

This is a prototype of serializer that respect this layout.
"""


from typing import Union
from typing import get_type_hints as gth

base_types = {int, str, bool, type(None)}


def serialize(instance, annotation):
    try:
        if annotation in base_types:
            # print("BASE", instance)
            assert isinstance(instance, annotation), f"{instance} {annotation}"
            return instance

        origin = getattr(annotation, "__origin__", None)
        if origin is list:
            assert isinstance(instance, origin), f"{instance} {origin}"
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
            serialized_data = serialize(instance, ann_)
            type_ = ann_.__name__
            if hasattr(ann_, "type"):
                type_ = ann_.type
            if isinstance(serialized_data, dict):
                return {**serialized_data, "type": type_}
            return {"data": serialized_data, "type": type_}
        if (
            (type(annotation) is type)
            and type.__module__ not in ("builtins", "typing")
            and (instance.__class__.__name__ == getattr(annotation, "__name__", None))
            or type(instance) == annotation
        ):
            data = {}
            type_ = type(instance).__name__
            if hasattr(instance, "type"):
                type_ = instance.type
            data["type"] = type_
            for k, ann in gth(type(instance)).items():
                data[k] = serialize(getattr(instance, k), ann)
            return data
    except Exception as e:
        e.add_note(f"serializing {instance.__class__}")
        raise
