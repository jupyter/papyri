"""
vestigial things from velin.
"""


import numpydoc.docscrape as nds
from numpydoc.docscrape import Parameter
from velin.examples_section_utils import reformat_example_lines

BLACK_REFORMAT = True


class NumpyDocString(nds.NumpyDocString):
    """
    subclass a littel bit more lenient on parsing
    """

    __slots__ = ()

    aliases = {
        "Parameters": (
            "options",
            "parameter",
            "parameters",
            "paramters",
            "parmeters",
            "paramerters",
            "arguments",
        ),
        "Yields": ("signals",),
    }

    def normalize(self):
        """
        Apply a bunch of heuristic that try to normalise the data.
        """
        if (params := self["Parameters"]) :
            for i, p in enumerate(params):
                if not p.type and ":" in p.name:
                    if p.name.startswith(".."):
                        continue
                    name, type_ = [_.strip() for _ in p.name.split(":", maxsplit=1)]
                    params[i] = nds.Parameter(name, type_, p[2])

    def parse_examples(self, lines, indent=4):
        # this is bad practice be we do want normalisation here for now
        # to  check that parse->format->parse is idempotent.
        # this can be done if we had a separate "normalize" step.
        global BLACK_REFORMAT
        if BLACK_REFORMAT:
            try:
                lines = reformat_example_lines(lines, indent=indent)
            except Exception:
                print("black failed")
                print("\n".join(lines))
                raise

        return lines

    def to_json(self):

        res = {k: v for (k, v) in self.__dict__.items() if ((k not in {"_doc"}) and v)}
        res["_parsed_data"] = {k: v for (k, v) in res["_parsed_data"].items() if v}

        return res

    @classmethod
    def from_json(cls, obj):
        nds = cls("")
        nds.__dict__.update(obj)
        # print(obj['_parsed_data'].keys())
        nds._parsed_data["Parameters"] = [
            Parameter(a, b, c) for (a, b, c) in nds._parsed_data.get("Parameters", [])
        ]

        for it in (
            "Returns",
            "Yields",
            "Extended Summary",
            "Receives",
            "Other Parameters",
            "Raises",
            "Warns",
            "Warnings",
            "See Also",
            "Notes",
            "References",
            "Examples",
            "Attributes",
            "Methods",
        ):
            if it not in nds._parsed_data:
                nds._parsed_data[it] = []
        for it in ("index",):
            if it not in nds._parsed_data:
                nds._parsed_data[it] = {}
        return nds

    def __init__(self, *args, **kwargs):
        self.ordered_sections = []
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in ["Extended Summary", "Summary"]:
            value = [d.rstrip() for d in value]

        if key in ("Examples"):
            value = self.parse_examples(value)
        super().__setitem__(key, value)
        assert (
            key not in self.ordered_sections
        ), f"assert {key} not in {self.ordered_sections}, {super().__getitem__(key)}, {value}"
        self.ordered_sections.append(key)

    def _guess_header(self, header):
        if header in self.sections:
            return header
        # handle missing trailing `s`, and trailing `:`
        for s in self.sections:
            if s.lower().startswith(header.rstrip(":").lower()):
                return s
        for k, v in self.aliases.items():
            if header.lower() in v:
                return k
        raise ValueError("Cound not find match for section:", header)

    def _read_sections(self):
        for name, data in super()._read_sections():
            name = self._guess_header(name)
            yield name, data

    def _parse_param_list(self, *args, **kwargs):
        """
        Normalize parameters
        """
        parms = super()._parse_param_list(*args, **kwargs)
        out = []
        for name, type_, desc in parms:
            out.append(
                nds.Parameter(name.strip(), type_.strip(), [d.rstrip() for d in desc])
            )
        return out
