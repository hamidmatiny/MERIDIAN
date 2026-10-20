"""Incident severity, assigned during triage."""

from enum import StrEnum


class Severity(StrEnum):
    """How urgently an incident must be handled. SEV1 is the most severe."""

    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"
