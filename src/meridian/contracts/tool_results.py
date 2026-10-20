"""Typed payloads returned by the read-only MCP tools.

Every response carries a stable ``ref`` the root-cause agent may cite.
"""

from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.change_event import ChangeEvent, EventKind
from meridian.contracts.validation import FiniteFloat, NonBlank, UtcDatetime


class LineageResult(MeridianModel):
    """Upstream or downstream datasets for one name."""

    ref: NonBlank = Field(description="Stable ref, lineage:upstream:<dataset> or downstream.")
    dataset: NonBlank = Field(description="Dataset that was queried.")
    direction: NonBlank = Field(description="upstream or downstream.")
    datasets: list[NonBlank] = Field(description="Related datasets, nearest first.")


class ContractSummary(MeridianModel):
    """Pandera column and check names for one dataset."""

    ref: NonBlank = Field(description="Stable ref, contract:<dataset>.")
    dataset: NonBlank = Field(description="Dataset whose contract was summarized.")
    contract_name: NonBlank = Field(description="Pandera model name.")
    columns: list[NonBlank] = Field(description="Declared column names.")
    checks: list[NonBlank] = Field(description="Named dataframe checks, if any.")


class ColumnDescription(MeridianModel):
    """Dtype and null rate for one column of a partition."""

    column: NonBlank = Field(description="Column name.")
    dtype: NonBlank = Field(description="Pandas dtype string.")
    null_rate: FiniteFloat = Field(ge=0, le=1, description="Fraction of nulls.")
    ref: NonBlank = Field(description="Stable ref, column:<path>:<column>.")


class PartitionDescription(MeridianModel):
    """Shape of one parquet partition."""

    ref: NonBlank = Field(description="Stable ref, partition:<path>.")
    path: NonBlank = Field(description="Parquet path that was read.")
    row_count: int = Field(ge=0, description="Number of rows.")
    columns: list[ColumnDescription] = Field(description="One entry per column.")


class ColumnComparison(MeridianModel):
    """PSI and null rate for one profiled column."""

    column: NonBlank = Field(description="Column name.")
    psi: float | None = Field(description="PSI against the reference profile. Unset if not finite.")
    null_rate: FiniteFloat = Field(ge=0, le=1, description="Fraction of nulls in the partition.")
    ref: NonBlank = Field(description="Stable ref, psi:<dataset>:<column>.")


class ReferenceComparison(MeridianModel):
    """Per-column comparison of a partition to a reference profile."""

    ref: NonBlank = Field(description="Stable ref, psi:<dataset>.")
    dataset: NonBlank = Field(description="Dataset named by the reference profile.")
    columns: list[ColumnComparison] = Field(description="One comparison per profiled column.")


class FailureSample(MeridianModel):
    """Sample rows stored on an incident."""

    ref: NonBlank = Field(description="Stable ref, sample:<incident_id>.")
    incident_id: UUID = Field(description="Incident the rows belong to.")
    rows: list[dict[str, str | int | float | bool | None]] = Field(
        description="Up to 20 scalar sample rows.",
    )


class EventView(MeridianModel):
    """One change event plus the ref an agent may cite."""

    id: UUID = Field(description="Event id.")
    ts: UtcDatetime = Field(description="UTC timestamp.")
    kind: EventKind = Field(description="Event kind.")
    dataset: NonBlank = Field(description="Dataset the event applies to.")
    author: NonBlank = Field(description="Author recorded on the event.")
    summary: NonBlank = Field(description="Event summary.")
    diff_excerpt: str = Field(description="Diff excerpt.")
    ref: NonBlank = Field(description="Stable ref, event:<id>.")


class EventPage(MeridianModel):
    """Events matching a history query."""

    ref: NonBlank = Field(description="Stable ref for this query result.")
    events: list[EventView] = Field(description="Matching events, each with its own ref.")


def view_event(event: ChangeEvent) -> EventView:
    """Copy a log record into the tool view, with its stable ref."""
    return EventView(
        id=event.id,
        ts=event.ts,
        kind=event.kind,
        dataset=event.dataset,
        author=event.author,
        summary=event.summary,
        diff_excerpt=event.diff_excerpt,
        ref=f"event:{event.id}",
    )
