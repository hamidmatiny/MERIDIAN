"""Quantile profile used by the distribution-shift detector."""

from itertools import pairwise
from typing import Self

from pydantic import Field, model_validator

from meridian.contracts.base import MeridianModel
from meridian.contracts.validation import FiniteFloat, NonBlank


class ColumnProfile(MeridianModel):
    """Equal-probability bins for one numeric column."""

    column: NonBlank = Field(description="Column these bins describe.")
    bin_edges: list[float] = Field(min_length=2, description="Inclusive histogram edges.")
    expected_proportions: list[float] = Field(
        min_length=1,
        description="Expected share of rows in each bin. One fewer entry than bin_edges.",
    )

    @model_validator(mode="after")
    def bins_match_proportions(self) -> Self:
        """Require one proportion per interval and strictly increasing edges."""
        if len(self.expected_proportions) != len(self.bin_edges) - 1:
            raise ValueError("expected_proportions must have one entry per bin")
        edges = self.bin_edges
        if any(right <= left for left, right in pairwise(edges)):
            raise ValueError("bin_edges must be strictly increasing")
        if any(share <= 0 for share in self.expected_proportions):
            raise ValueError("expected_proportions must be positive")
        return self


class ReferenceProfile(MeridianModel):
    """Stored quantile profile for one dataset."""

    dataset: NonBlank = Field(description="Dataset this profile describes.")
    psi_threshold: FiniteFloat = Field(
        gt=0,
        description="Population Stability Index at or above this value is a shift.",
    )
    columns: list[ColumnProfile] = Field(
        min_length=1,
        description="Numeric columns compared against the profile.",
    )
