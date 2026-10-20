"""Deterministic replay of failing data against a patch."""

from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from meridian.contracts.base import MeridianModel


class VerificationResult(MeridianModel):
    """Pass/fail from a data replay. No model judgment belongs in this record."""

    patch_id: UUID = Field(description="Patch that was replayed.")
    passed: bool = Field(
        description=(
            "True only when failures_remaining is 0. "
            "False only when failures_remaining is greater than 0."
        ),
    )
    rows_checked: int = Field(ge=0, description="Number of rows the verifier replayed.")
    failures_remaining: int = Field(
        ge=0,
        description=(
            "Contract failures still present after the patch. "
            "Must be 0 when passed is true and greater than 0 when passed is false."
        ),
    )
    duration_ms: int = Field(
        ge=0, description="Wall-clock duration of the replay, in milliseconds."
    )

    @model_validator(mode="after")
    def passed_matches_remaining_failures(self) -> Self:
        """Pass and fail must agree with the remaining failure count."""
        if self.passed and self.failures_remaining != 0:
            raise ValueError("passed verification requires failures_remaining == 0")
        if not self.passed and self.failures_remaining == 0:
            raise ValueError("failed verification requires failures_remaining > 0")
        return self
