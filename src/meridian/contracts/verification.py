"""Deterministic replay of failing data against a patch."""

from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from meridian.contracts.base import MeridianModel


class VerificationResult(MeridianModel):
    """Pass/fail from a data replay. No model judgment belongs in this record."""

    patch_id: UUID = Field(description="Patch that was replayed.")
    passed: bool = Field(description="True only when no contract failures remain.")
    rows_checked: int = Field(ge=0, description="Number of rows the verifier replayed.")
    failures_remaining: int = Field(
        ge=0,
        description="Contract failures still present after the patch.",
    )
    duration_ms: int = Field(
        ge=0, description="Wall-clock duration of the replay, in milliseconds."
    )

    @model_validator(mode="after")
    def passing_result_has_no_failures(self) -> Self:
        """A pass cannot report leftover contract failures."""
        if self.passed and self.failures_remaining != 0:
            raise ValueError(
                "passed verification cannot leave failures_remaining greater than zero"
            )
        return self
