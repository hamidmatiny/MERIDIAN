"""Shared validators for contract fields."""

import math
from datetime import UTC, datetime, timedelta
from typing import Annotated

from pydantic import AfterValidator, BeforeValidator


def require_utc(value: datetime) -> datetime:
    """Reject naive or non-UTC timestamps and normalize to ``timezone.utc``."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("timestamp must be timezone-aware UTC, not naive")
    offset = value.utcoffset()
    if offset != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return value.astimezone(UTC)


def require_non_blank(value: str) -> str:
    """Reject empty or whitespace-only strings."""
    if value.strip() == "":
        raise ValueError("must not be blank")
    return value


def as_finite_float(value: object) -> float:
    """Accept an int or float. Reject booleans, NaN, and infinities."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("must be a finite number")
    if not math.isfinite(value):
        raise ValueError("must be a finite number")
    return float(value)


UtcDatetime = Annotated[datetime, AfterValidator(require_utc)]
NonBlank = Annotated[str, AfterValidator(require_non_blank)]
FiniteFloat = Annotated[float, BeforeValidator(as_finite_float)]
