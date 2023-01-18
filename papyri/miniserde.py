"""

A mini-implementation of an automatic serialiser-deserialiser for nested
dataclass like class based on type annotations.

Example::


    In [14]: from dataclasses import dataclass
        ...: from typing import Optional, Union, List
        ...:

    Note that Author and Reviewer are isomorphic even if totally unrelated.

    In [15]: @dataclass
        ...: class Author:
        ...:     first: Optional[str]
        ...:     last: str
        ...:
        ...: @dataclass
        ...: class Reviewer:
        ...:     first: Optional[str]
        ...:     last: str
        ...:

    Here, items can be heterogenous, or of ambiguous type based only on its fields values.

    In [16]: @dataclass
        ...: class Book:
        ...:     author: List[Union[Author, Reviewer]]
        ...:     title: str
        ...:


    In [17]: obj = Book([Author("Matthias", "B"), Reviewer("Tony", "Fast")], "pyshs")
        ...:
        ...: data = serialize(obj , Book)
        ...:
        ...: deserialize(Book, Book, data)

    Out[17]: Book(author=[Author(first='Matthias', last='B'), Reviewer(first='Tony', last='Fast')], title='pyshs')

                          ^...................................^
                                            .
                                            .Note the conserved types.



Unlike other similar libraries that automatically serialise/deserialise it has
the following properties:

- object do not need to have a give baseclass, they need to have an __init__
  or _deserialise class method that takes each parameter as kwargs.
- Subclass or isomorphic classes are kept in the de-serialisation, in
  particular in Union and List of Unions. That is to say it will properly
  de-serialise and heterogenous list or dict, as long as those respect the
  type annotation.

Both Pydantic and Jetblack-serialize would have erased the types and returned
either 2 Authors or 2 Reviewers.

- it is also compatible with Rust Serde with adjacently tagged Unions (not
      critical but nice to have)

"""


from functools import lru_cache
from typing import Union
from typing import get_type_hints as gth

base_types = {int, str, bool, type(None)}


@lru_cache(50)
def get_type_hints(type_):
    return gth(type_)


def serialize(instance, annotation):
    exception_already_desribed = False
    try:
        if (annotation in base_types) and (isinstance(instance, annotation)):
            return instance
        elif getattr(annotation, "__origin__", None) is tuple and isinstance(
            instance, tuple
        ):
            # this may be slightly incorrect as usually tuple as positionally type dependant.
            inner_annotation = annotation.__args__
            # assert len(inner_annotation) == 1, inner_annotation
            return tuple(serialize(x, inner_annotation[0]) for x in instance)
        elif getattr(annotation, "__origin__", None) is list and isinstance(
            instance, list
        ):
            inner_annotation = annotation.__args__
            # assert len(inner_annotation) == 1, inner_annotation
            return [serialize(x, inner_annotation[0]) for x in instance]
        elif getattr(annotation, "__origin__", None) is dict:
            # assert type(instance) == dict
            key_annotation, value_annotation = annotation.__args__
            # assert key_annotation == str, key_annotation
            return {k: serialize(v, value_annotation) for k, v in instance.items()}

        elif getattr(annotation, "__origin__", None) is Union:
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
        elif (
            (type(annotation) is type)
            and type.__module__ not in ("builtins", "typing")
            and (instance.__class__.__name__ == getattr(annotation, "_name", None))
            or type(instance) == annotation
        ):
            if hasattr(instance, "_validate"):
                instance._validate()
            data = {}
            for k, v in get_type_hints(type(instance)).items():
                try:
                    data[k] = serialize(getattr(instance, k), v)
                except Exception as e:
                    exception_already_desribed = True
                    raise type(e)(f"Error serializing field {k!r} of {instance!r}")
            return data

        else:
            assert False, (
                f"Error serializing {instance!r}\n, of type {type(instance)!r} "
                f"expected  {annotation}, got {type(instance)}"
            )
    except Exception as e:
        if exception_already_desribed:
            raise
        raise type(e)(
            f"Error serialising {instance!r}, of type {type(instance)} "
            f"expecting {annotation}, got {type(instance)}"
        ) from e


# type_ and annotation are _likely_ duplicate here as an annotation is likely a type, or  a List, Union, ....)
def deserialize(type_, annotation, data):
    # assert type_ is annotation
    # assert annotation != {}
    # assert annotation is not dict
    # assert annotation is not None, "None is handled by nullable types"
    if annotation is str:
        # assert isinstance(data, str)
        return data
    if annotation is int:
        # assert isinstance(data, int)
        return data
    if annotation is bool:
        # assert isinstance(data, bool)
        return data
    orig = getattr(annotation, "__origin__", None)
    if orig:
        if orig is tuple:
            # assert isinstance(data, list)
            inner_annotation = annotation.__args__
            # assert len(inner_annotation) == 1, inner_annotation
            return tuple(
                deserialize(inner_annotation[0], inner_annotation[0], x) for x in data
            )
        elif orig is list:
            # assert isinstance(data, list)
            inner_annotation = annotation.__args__
            # assert len(inner_annotation) == 1, inner_annotation
            return [
                deserialize(inner_annotation[0], inner_annotation[0], x) for x in data
            ]
        elif orig is dict:
            # assert isinstance(data, dict)
            _, value_annotation = annotation.__args__
            return {
                k: deserialize(value_annotation, value_annotation, x)
                for k, x in data.items()
            }
        elif orig is Union:
            inner_annotation = annotation.__args__
            if len(inner_annotation) == 2 and inner_annotation[1] == type(None):
                # assert inner_annotation[0] is not None
                if data is None:
                    return None
                else:
                    return deserialize(inner_annotation[0], inner_annotation[0], data)
            real_type = [t for t in inner_annotation if t.__name__ == data["type"]]
            # assert len(real_type) == 1, real_type
            real_type = real_type[0]
            return deserialize(real_type, real_type, data["data"])
        else:
            assert False
    elif (type(annotation) is type) and annotation.__module__ not in (
        "builtins",
        "typing",
    ):
        loc = {}
        new_ann = get_type_hints(annotation).items()
        # assert new_ann
        for k, v in new_ann:
            # assert k in data.keys(), f"{k} not int {data.keys()}"
            # if data[k] != 0:
            #     assert data[k] != {}, f"{data}, {k}"
            intermediate = deserialize(v, v, data[k])
            # assert intermediate != {}, f"{v}, {data}, {k}"
            loc[k] = intermediate
        if hasattr(annotation, "_deserialise"):
            return annotation._deserialise(**loc)
        else:
            return annotation(**loc)

    else:
        assert False, f"{annotation!r}, {data}"
