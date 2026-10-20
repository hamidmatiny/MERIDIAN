"""Deterministic dry-run of a remediation, shown to the critic before it spends tokens."""

from pydantic import Field

from meridian.contracts.base import MeridianModel


class DryRunReport(MeridianModel):
    """What a 200-row sandbox sample did, measured without a model."""

    rows_sampled: int = Field(ge=0, description="Rows in the dry-run sample.")
    rows_changed: int = Field(ge=0, description="Rows the remediation changed or dropped.")
    rows_changed_outside: int = Field(
        ge=0,
        description="Changed rows that were not in the incident's failing set.",
    )
    over_broad: bool = Field(
        description="True when the remediation modified rows outside the failing set.",
    )
    contracts_after: int = Field(ge=0, description="Incidents remaining on the sample afterward.")
    psi_after: str = Field(
        default="",
        description="PSI result on the sample after the remediation.",
    )
    detail: str = Field(default="", description="Extra context for the critic. Empty when unused.")
