"""MERIDIAN autonomous data-incident response.

Phase 0 exposes typed contracts and environment settings. The agent graph is planned.
"""

from meridian.config import MeridianConfigError, Settings, get_settings
from meridian.contracts import (
    MAX_CRITIC_ROUNDS,
    MAX_FIX_ATTEMPTS,
    ApprovalDecision,
    ApprovalOutcome,
    Evidence,
    FaultType,
    Hypothesis,
    Incident,
    MeridianState,
    Patch,
    Postmortem,
    Severity,
    TimelineEvent,
    Verdict,
    VerificationResult,
)

__version__ = "0.1.0"

__all__ = [
    "MAX_CRITIC_ROUNDS",
    "MAX_FIX_ATTEMPTS",
    "ApprovalDecision",
    "ApprovalOutcome",
    "Evidence",
    "FaultType",
    "Hypothesis",
    "Incident",
    "MeridianConfigError",
    "MeridianState",
    "Patch",
    "Postmortem",
    "Settings",
    "Severity",
    "TimelineEvent",
    "Verdict",
    "VerificationResult",
    "__version__",
    "get_settings",
]
