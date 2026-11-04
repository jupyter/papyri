import inspect
import json
import re
from dataclasses import dataclass
from typing import Any

try:
    import annotationlib
except ImportError:
    annotationlib = None  # type: ignore[assignment]

from .errors import TextSignatureParsingFailed
from .node_base import Node, register


@register(4031)
class Empty(Node):
    """Sentinel for an absent parameter annotation or default value.

    Mirrors ``inspect._empty``.  A ``SigParam`` field that holds ``Empty``
    means "not provided" — distinct from the annotation or default being the
    Python value ``None``.  The renderer must never display ``Empty``
    directly; use it only to detect absence.
    """

    pass


_empty = Empty()

NoneType = type(None)


@register(4030)
@dataclass
class SigParam(Node):
    """One parameter in a callable signature.

    ``annotation`` and ``default`` are three-valued:
    - ``str`` — the annotation/default as a string.
    - ``None`` — the annotation or default is the Python value ``None``.
    - ``Empty`` — no annotation or no default was provided.

    ``kind`` is the ``inspect._ParameterKind`` name, e.g.
    ``"POSITIONAL_OR_KEYWORD"``.
    """

    name: str
    # we likely want to make sure annotation is a structured object in the long run
    annotation: str | NoneType | Empty
    kind: str
    default: str | NoneType | Empty

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def to_parameter(self) -> inspect.Parameter:
        return inspect.Parameter(
            name=self.name,
            kind=getattr(inspect._ParameterKind, self.kind),
            default=inspect._empty if isinstance(self.default, Empty) else self.default,
            annotation=(
                inspect._empty
                if isinstance(self.annotation, Empty)
                else self.annotation
            ),
        )


@register(4029)
class SignatureNode(Node):
    """Structured callable signature extracted from ``inspect.signature``.

    ``kind`` classifies the callable: ``"function"``, ``"coroutine function"``,
    ``"generator function"``, ``"async_generator function"``, or
    ``"built-in function"``.  ``parameters`` preserves declaration order.
    ``return_annotation`` is ``Empty`` when no return annotation was given.
    """

    kind: str  # maybe enum, is it a function, async generator, generator, etc.
    parameters: tuple[SigParam, ...]  # of pairs, we don't use dict because of ordering
    return_annotation: Empty | str
    target_name: str
    type = "signature"

    def to_signature(self) -> inspect.Signature:
        return inspect.Signature([p.to_parameter() for p in self.parameters])


def clean_hexaddress(s: str) -> str:
    new = re.sub("0x[0-9a-f]+", "0x0000", s)
    return new


class Signature:
    """A wrapper around inspect utilities."""

    @classmethod
    def from_str(cls, sig: str, /) -> "Signature":
        """
        Create signature from a string version.

        Of course this is slightly incorrect as all the isgenerator and CO are going to wrong

        """
        glob: dict[str, Any] = {}
        oname = sig.split("(")[0]
        toexec = f"def {sig}:pass"
        try:
            exec(toexec, {}, glob)
            return cls(glob[oname])
        except Exception as e:
            raise TextSignatureParsingFailed(f"Unable to parse {toexec}") from e

    def __init__(self, target_item: Any) -> None:
        """
        Initialize the class.

        Parameters
        ----------
        target_item : callable
            The target item to be assigned.

        """
        self.target_item = target_item
        if annotationlib is not None:
            self._sig = inspect.signature(
                target_item,
                annotation_format=annotationlib.Format.STRING,
            )
        else:
            # Python 3.13 fallback: inspect.signature returns evaluated
            # annotations. Format them to strings to match 3.14 behavior.
            raw = inspect.signature(target_item)
            params = [
                p.replace(
                    annotation=inspect.formatannotation(p.annotation)
                    if p.annotation is not inspect.Parameter.empty
                    else p.annotation,
                )
                for p in raw.parameters.values()
            ]
            return_annotation = (
                inspect.formatannotation(raw.return_annotation)
                if raw.return_annotation is not inspect.Signature.empty
                else raw.return_annotation
            )
            self._sig = raw.replace(
                parameters=params, return_annotation=return_annotation
            )

    def to_node(self) -> SignatureNode:
        kind = ""
        if inspect.isfunction(self.target_item):
            kind = "function"
        if inspect.isbuiltin(self.target_item):
            kind = "built-in function"
        if inspect.isgeneratorfunction(self.target_item):
            kind = "generator function"
        if inspect.isasyncgenfunction(self.target_item):
            kind = "async_generator function"
        if inspect.iscoroutinefunction(self.target_item):
            kind = "coroutine function"
        if kind == "":
            kind = "function"
            # TODO: fix this, things like numpy's histogram2d's are weird
            # assert False, f"Unknown kind for {self.target_item}"

        # Why do we want to make sure this is not a coroutine?
        # What is special about a coroutine in this context?
        assert not inspect.iscoroutine(self.target_item)

        parameters = []
        for param in self.parameters.values():
            annotation: Empty | str
            if param.annotation is inspect._empty:
                annotation = _empty
            elif isinstance(param.annotation, str):
                annotation = param.annotation
            else:
                # TODO: Keep the original annotation object somewhere
                annotation = clean_hexaddress(
                    inspect.formatannotation(param.annotation)
                )
            parameters.append(
                SigParam(
                    name=param.name,
                    annotation=annotation,
                    kind=param.kind.name,
                    default=(
                        _empty
                        if param.default is inspect._empty
                        else clean_hexaddress(repr(param.default))
                    ),
                )
            )
        assert isinstance(kind, str)
        return SignatureNode(
            kind=kind,
            parameters=parameters,
            return_annotation=(
                _empty
                if self._sig.return_annotation is inspect._empty
                else str(self._sig.return_annotation)
            ),
            target_name=self.target_item.__name__,
        )

    @property
    def parameters(self) -> Any:
        return self._sig.parameters

    def param_default(self, param: str) -> Any:
        return self.parameters.get(param).default

    @property
    def annotations(self) -> dict[str, Any]:
        return self.target_item.__annotations__  # type: ignore[no-any-return]

    @property
    def return_annotation(self) -> Empty | str:
        return_annotation = self._sig.return_annotation
        return (
            _empty
            if return_annotation is inspect._empty
            else inspect.formatannotation(return_annotation)
        )

    @property
    def is_public(self) -> bool:
        return not self.target_item.__name__.startswith("_")

    @property
    def positional_only_parameter_count(self) -> int | None:
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
    def keyword_only_parameter_count(self) -> int | None:
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

    def to_dict(self) -> dict[str, Any]:
        """
        Output self as JSON (Python dict), using the same format as Griffe
        """
        json_data = self.to_node().to_dict()

        # Use human-readable names for parameter kinds
        for param in json_data["parameters"]:
            param["kind"] = getattr(inspect._ParameterKind, param["kind"]).description

        json_data["returns"] = self.return_annotation

        return json_data

    def to_json(self) -> bytes:
        """
        Output self as JSON, using the same format as Griffe
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True).encode()

    def __str__(self) -> str:
        return str(self._sig)
