import inspect

from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union
from .common_ast import Node, validate
from .errors import TextSignatureParsingFailed

from .common_ast import register


@register(4031)
class Empty(Node):
    pass


_empty = Empty()

NoneType = type(None)


@register(4030)
@dataclass
class ParameterNode(Node):
    name: str
    # we likely want to make sure annotation is a structured object in the long run
    annotation: Union[str, NoneType, Empty]
    kind: str
    default: Union[str, NoneType, Empty]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_parameter(self) -> inspect.Parameter:
        return inspect.Parameter(
            name=self.name,
            kind=getattr(inspect._ParameterKind, self.kind),
            default=inspect._empty if isinstance(self.default, Empty) else None,
            annotation=inspect._empty if isinstance(self.annotation, Empty) else self.annotation,
        )

@register(4029)
class SignatureNode(Node):
    kind: str  # maybe enum, is it a function, async generator, generator, etc.
    parameters: List[ParameterNode]  # of pairs, we don't use dict because of ordering

    def to_signature(self):
        return inspect.Signature([p.to_parameter() for p in self.parameters])


class Signature:
    """A wrapper around inspect utilities."""

    @classmethod
    def from_str(cls, sig: str, /) -> "Signature":
        """
        Create signature from a string version.

        Of course this is slightly incorrect as all the isgerator and CO are going to wrong

        """
        glob: Dict[str, Any] = {}
        oname = sig.split("(")[0]
        toexec = f"def {sig}:pass"
        try:
            exec(toexec, {}, glob)
        except Exception as e:
            raise TextSignatureParsingFailed(f"Unable to parse {toexec}") from e
        return cls(glob[oname])

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

    def to_node(self) -> SignatureNode:
        if inspect.isbuiltin(self.target_item):
            kind = "builtins"
        elif inspect.isfunction(self.target_item):
            kind = "function"
        elif self.is_generator:
            kind = "generator"
        elif self.is_async_generator:
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
                    name=param.name,
                    annotation=_empty if param.annotation is inspect._empty else str(param.annotation),
                    kind=param.kind.name,
                    default=_empty if param.default is inspect._empty else str(param.default),
                )
            )
        assert isinstance(kind, str)
        return SignatureNode(kind=kind, parameters=parameters)

    @property
    def parameters(self):
        return self._sig.parameters

    @property
    def is_async_function(self) -> bool:
        return inspect.iscoroutinefunction(self.target_item)

    @property
    def is_async_generator(self) -> bool:
        return inspect.isasyncgenfunction(self.target_item)

    @property
    def is_generator(self) -> bool:
        return inspect.isgenerator(self.target_item)

    def param_default(self, param):
        return self.parameters.get(param).default

    @property
    def annotations(self) -> bool:
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
