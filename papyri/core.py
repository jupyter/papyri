"""
Core Papyri data structures. 

This should likely be the most stable part of Papyri as it is what handles and validate the intermediate
representation(s)

It should likely be the modules with the less dependencies as well as being synchronous, to be usable from most context
and minimal installs.
"""


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
    see_also = None  # see also data
    edata = None  # example data
    refs = None  # references
    # keys and values of all the sections.
    content = None
    version = None  # version of current package

    def __init__(self, ndoc):
        self.see_also = ndoc.see_also

        # todo: shoudl always have a edata attr.
        self.edata = getattr(ndoc, "edata", None)
        self.refs = ndoc.refs
        self.content = {}
        self.version = ndoc.version
        for k, v in ndoc.items():
            self.content[k] = v
