import inspect


from dataclasses import dataclass
from typing import Optional, List, Tuple
from .common_ast import Node


@dataclass
class ParameterNode(Node):
    name: str
    # we likely want to make sure annotation is a strutured object in the long run
    annotation: Optional[str]
    kind: str
    default: Optional[str]

    def to_parameter(self):
        return inspect.Parameter(
            name=self.name,
            kind=getattr(inspect._ParameterKind, self.kind),
            default=self.default,
            annotation=self.annotation,
        )


class SignatureNode(Node):
    kind: str  # maybe enum, is it a function, async generator, generator, etc.
    parameters: List[ParameterNode]  # of pairs, we don't use dict because of orderin


class Signature:
    """A wrapper around inspect utilities."""

    def __init__(self, target_item):
        """
        Initialize the class.

        Parameters
        ----------
        target_item : callable
            The target item to be assigned.

        """
        self.target_item = target_item
        self._sig = inspect.signature(target_item)

    def to_node(self):
        if inspect.isfunction(self.target_item):
            kind = "function"
        elif self.is_generator():
            kind = "generator"
        elif self.is_async_generator():
            kind = "async_generator"
        elif inspect.iscoroutinefunction(self.target_item):
            kind = "coroutine_function"
        else:
            assert False, f"Unknown kind for {self.target_item}"
        assert not inspect.iscoroutine(self.target_item)

        parameters = []
        for param in self.parameters.values():
            parameters.append(
                ParameterNode(
                    param.name,
                    None if param.annotation is inspect._empty else param.annotation,
                    param.kind.name,
                    None if param.default else str(param.default),
                )
            )

        return SignatureNode(kind, parameters)

    @property
    def parameters(self):
        return self._sig.parameters

    @property
    def is_async_function(self):
        return inspect.iscoroutinefunction(self.target_item)

    @property
    def is_async_generator(self):
        return inspect.isasyncgenfunction(self.target_item)

    @property
    def is_generator(self):
        return inspect.isgenerator(self.target_item)

    def param_default(self, param):
        return self.parameters.get(param).default

    @property
    def annotations(self):
        return self.target_item.__annotations__

    @property
    def is_public(self) -> bool:
        return not self.target_item.__name__.startswith("_")

    @property
    def positional_only_parameter_count(self):
        """Number of positional-only parameters in a signature.
        `None` if `obj` has no signature.
        """
        if self._sig:
            return sum(
                1
                for p in self.parameters.values()
                if p.kind is inspect.Parameter.POSITIONAL_ONLY
            )
        else:
            return None

    @property
    def keyword_only_parameter_count(self):
        """Number of keyword-only parameters in a signature.
        `None` if `obj` has no signature.
        """
        if self._sig:
            return sum(
                1
                for p in self.parameters.values()
                if p.kind is inspect.Parameter.KEYWORD_ONLY
            )
        else:
            return None

    def __str__(self):
        return str(self._sig)
