"""Incident severity, assigned during triage."""

from enum import StrEnum


class Severity(StrEnum):
    """How urgently an incident must be handled. SEV1 is the most severe."""

    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"


# Higher rank is more severe. SEV1 outranks SEV4.
SEVERITY_RANK: dict[Severity, int] = {
    Severity.SEV4: 1,
    Severity.SEV3: 2,
    Severity.SEV2: 3,
    Severity.SEV1: 4,
}


def more_severe(left: Severity, right: Severity) -> Severity:
    """Return the more severe of two levels. SEV1 wins over SEV4."""
    if SEVERITY_RANK[left] >= SEVERITY_RANK[right]:
        return left
    return right
