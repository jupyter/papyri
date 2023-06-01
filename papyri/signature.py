import inspect


class Signature:
    """A wrapper around inspect utilities."""

    def __init__(self, target_item):
        """
        :param target_item: A callable
        """
        self.target_item = target_item
        self._sig = inspect.signature(target_item)

    @property
    def parameters(self):
        return self._sig.parameters

    @property
    def is_async(self):
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
