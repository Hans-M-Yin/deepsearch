"""Explicit rejection types distinguish content decisions from service errors."""


class ReviewRejected(ValueError):
    pass


class StructureRejected(ValueError):
    pass


class SkipAugmentation(ValueError):
    """Expected policy skip, preserving the original record."""

    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}
