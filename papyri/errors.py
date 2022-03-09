class IncorrectInternalDocsLen(AssertionError):
    pass

class NumpydocParseError(ValueError):
    pass

class ExampleError1(ValueError):
    pass

class StrictParsingError(Exception):
    pass


class SpaceAfterBlockDirectiveError(Exception):
    pass


class VisitSubstitutionDefinitionNotImplementedError(NotImplementedError):
    pass


class VisitCitationReferenceNotImplementedError(NotImplementedError):
    pass


class VisitCitationNotImplementedError(NotImplementedError):
    pass


class SerialisationError(Exception):
    pass
