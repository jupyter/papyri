"""
Core Papyri data structures. 

This should likely be the most stable part of Papyri as it is what handles and validate the intermediate
representation(s)

It should likely be the modules with the less dependencies as well as being synchronous, to be usable from most context
and minimal installs.
"""
from __future__ import annotations

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "__to_json__"):
            return o.__to_json__(self)
        elif dataclass.is_dataclass(o):
            return dataclass.asdict(o)
        return super().default(o)

    def decode(self, s):
        json.loads(s, object_hook=self.hook)


class DocData:
    """
    Represent the (in-memory) structure of an object documentation.
    As well as way to reliably serialise/deserialise it.

    TBD:
        - how should external resources like images be loaded ?

    """

    sections = [
        "Signature",
        "Summary",
        "Extended Summary",
        "Parameters",
        "Returns",
        "Yields",
        "Receives",
        "Raises",
        "Warns",
        "Other Parameters",
        "Attributes",
        "Methods",
        "See Also",
        "Notes",
        "Warnings",
        "References",
        "Examples",
        "index",
    ]  # List of sections in order
    see_also: List[SeeAlsoItem]  # see also data
    edata = None  # example data
    refs = None  # references
    # keys and values of all the sections.
    content = None
    version = None  # version of current package

    def __init__(self, doc_blob):
        assert hasattr(doc_blob, "see_also")
        self.see_also = doc_blob.see_also

        self.example_section_data = doc_blob.example_section_data
        self.refs = doc_blob.refs
        self.content = doc_blob.content
        self.version = doc_blob.version
        # for k, v in doc_blob.sections.items():
        #    self.content[k] = v


@dataclass
class Ref:
    name: str
    ref: Optional[str]
    exists: Optional[bool]

    def __hash__(self):
        return hash((self.name, self.ref, self.exists))


@dataclass
class SeeAlsoItem:
    name: Ref
    descriptions: str
    # there are a few case when the lhs is `:func:something`... in scipy.
    type: str

    @classmethod
    def from_json(cls, name, descriptions, type):
        return cls(Ref(**name), descriptions, type)

    def __hash__(self):
        return hash((self.name, tuple(self.descriptions)))
