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
    descriptions: List[Any]
    # there are a few case when the lhs is `:func:something`... in scipy.
    type: str

    @classmethod
    def from_json(cls, name, descriptions, type):
        return cls(Ref(**name), descriptions, type)

    def __hash__(self):
        return hash((self.name, tuple(self.descriptions)))
