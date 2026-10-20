"""What the grounding guardrail removed, and how event ids are compared."""

from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import NonBlank

_SAMPLE_LIMIT = 8


class GroundingViolation(MeridianModel):
    """One citation the guardrail rejected."""

    field: NonBlank = Field(description="Hypothesis field that was checked, such as evidence.")
    raw: NonBlank = Field(description="Value the model returned, before normalization.")
    normalized: NonBlank = Field(description="Form the value was compared in.")
    allowed_refs_sample: list[NonBlank] = Field(
        default_factory=list,
        description="A short sample of the refs this case's tools returned.",
    )


def event_key(value: str) -> str | None:
    """Return the UUID inside ``event:<id>`` or a bare id. Other strings are not events."""
    text = value.strip()
    if text.startswith("event:"):
        text = text[len("event:") :]
    try:
        return str(UUID(text))
    except ValueError:
        return None


def canonical_event_ref(value: str) -> str | None:
    """Return ``event:<uuid>`` when ``value`` names an event, otherwise None."""
    key = event_key(value)
    if key is None:
        return None
    return f"event:{key}"


def event_uuid(value: str) -> UUID | None:
    """Parse an event id. ``event:<id>`` and ``<id>`` are the same event."""
    key = event_key(value)
    if key is None:
        return None
    return UUID(key)


def ref_sample(refs: set[str]) -> list[str]:
    """Stable short sample of a ref set, for a violation record."""
    return sorted(refs)[:_SAMPLE_LIMIT]
