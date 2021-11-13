class StrictParsingError(Exception):
    pass


class SpaceAfterBlockDirectiveError(Exception):
    pass


class VisitSubstitutionDefinitionNotImplementedError(NotImplementedError):
    pass


class VisitTargetNotImplementedError(NotImplementedError):
    pass


class VisitCommentNotImplementedError(NotImplementedError):
    pass


class VisitCitationReferenceNotImplementedError(NotImplementedError):
    pass


class VisitCitationNotImplementedError(NotImplementedError):
    pass


class SerialisationError(Exception):
    pass
