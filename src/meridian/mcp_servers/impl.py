"""Read-only tool implementations shared by the MCP servers."""

import math
import os
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pandas as pd

from meridian.contracts.change_event import ChangeEvent, EventKind
from meridian.contracts.incident import Incident
from meridian.contracts.reference_profile import ReferenceProfile
from meridian.contracts.tool_results import (
    ColumnComparison,
    ColumnDescription,
    ContractSummary,
    EventPage,
    EventView,
    FailureSample,
    LineageResult,
    PartitionDescription,
    ReferenceComparison,
    view_event,
)
from meridian.injector.events import load_events
from meridian.pipeline.lineage import downstream_of, upstream_of
from meridian.pipeline.parquet import read_parquet
from meridian.quality.contracts import model_for
from meridian.quality.psi import population_stability_index


def lineage_result(dataset: str, *, direction: str) -> LineageResult:
    """Upstream or downstream names for ``dataset``."""
    if direction == "upstream":
        names = upstream_of(dataset)
        ref = f"lineage:upstream:{dataset}"
    elif direction == "downstream":
        names = downstream_of(dataset)
        ref = f"lineage:downstream:{dataset}"
    else:
        raise ValueError("direction must be upstream or downstream")
    return LineageResult(ref=ref, dataset=dataset, direction=direction, datasets=names)


def contract_summary(dataset: str) -> ContractSummary:
    """Column names and named dataframe checks from the pandera model."""
    model = model_for(dataset)
    schema = model.to_schema()
    raw_checks = getattr(schema, "checks", [])
    checks: list[str] = []
    if isinstance(raw_checks, list):
        for check in raw_checks:
            name = getattr(check, "name", None)
            if isinstance(name, str) and name != "":
                checks.append(name)
    config = getattr(model, "Config", None)
    contract_name = str(getattr(config, "name", model.__name__))
    return ContractSummary(
        ref=f"contract:{dataset}",
        dataset=dataset,
        contract_name=contract_name,
        columns=[str(column) for column in schema.columns],
        checks=checks,
    )


def describe_partition(path: str) -> PartitionDescription:
    """Dtypes and null rates for every column in a parquet file."""
    frame = read_parquet(Path(path))
    columns = [
        ColumnDescription(
            column=str(name),
            dtype=str(frame[name].dtype),
            null_rate=_null_rate(frame[name]),
            ref=f"column:{path}:{name}",
        )
        for name in frame.columns
    ]
    return PartitionDescription(
        ref=f"partition:{path}",
        path=path,
        row_count=len(frame),
        columns=columns,
    )


def compare_to_reference(path: str, profile_path: str) -> ReferenceComparison:
    """PSI and null rate for each column in the reference profile."""
    frame = read_parquet(Path(path))
    profile = ReferenceProfile.model_validate_json(Path(profile_path).read_text(encoding="utf-8"))
    columns: list[ColumnComparison] = []
    for column in profile.columns:
        if column.column in frame.columns:
            psi_value = population_stability_index(
                frame[column.column].to_numpy(dtype=float),
                column,
            )
            psi = None if not math.isfinite(psi_value) else psi_value
            null_rate = _null_rate(frame[column.column])
        else:
            psi = None
            null_rate = 1.0
        columns.append(
            ColumnComparison(
                column=column.column,
                psi=psi,
                null_rate=null_rate,
                ref=f"psi:{profile.dataset}:{column.column}",
            )
        )
    return ReferenceComparison(
        ref=f"psi:{profile.dataset}",
        dataset=profile.dataset,
        columns=columns,
    )


def sample_failures(root: Path, incident_id: str, n: int) -> FailureSample:
    """Return up to ``n`` sample rows stored for the incident. ``n`` is at most 20."""
    if n < 1 or n > 20:
        raise ValueError("n must be from 1 through 20")
    path = root / "incidents" / f"{incident_id}.json"
    incident = Incident.model_validate_json(path.read_text(encoding="utf-8"))
    if str(incident.id) != incident_id:
        raise ValueError("incident file id does not match the request")
    return FailureSample(
        ref=f"sample:{incident.id}",
        incident_id=incident.id,
        rows=list(incident.sample_failure_rows[:n]),
    )


def list_change_events(
    root: Path,
    *,
    dataset: str | None = None,
    since: str | None = None,
    until: str | None = None,
    kind: str | None = None,
) -> EventPage:
    """Filter ``events.jsonl``. Every event keeps its ``event:<id>`` ref."""
    events = load_events(root / "events.jsonl")
    since_at = _parse_time(since)
    until_at = _parse_time(until)
    kind_value = None if kind is None else EventKind(kind)
    chosen = [
        event
        for event in events
        if _matches(event, dataset=dataset, since=since_at, until=until_at, kind=kind_value)
    ]
    label = dataset or "all"
    return EventPage(ref=f"events:{label}", events=[view_event(event) for event in chosen])


def get_change_event(root: Path, event_id: str) -> EventView:
    """Return one event. Raises ``KeyError`` when the id is absent."""
    target = UUID(event_id)
    for event in load_events(root / "events.jsonl"):
        if event.id == target:
            return view_event(event)
    raise KeyError(f"unknown event {event_id}")


def publish_incident(root: Path, incident: Incident) -> None:
    """Write the incident where ``sample_failures`` can read it."""
    dest = root / "incidents"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{incident.id}.json").write_text(incident.model_dump_json(), encoding="utf-8")


def evidence_root_from_env() -> Path:
    """Directory the stdio servers read. Set ``MERIDIAN_EVIDENCE_ROOT``."""
    raw = os.environ.get("MERIDIAN_EVIDENCE_ROOT")
    if raw is None or raw.strip() == "":
        raise RuntimeError("MERIDIAN_EVIDENCE_ROOT is not set")
    return Path(raw)


def _null_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def _parse_time(value: str | None) -> datetime | None:
    if value is None or value.strip() == "":
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("since and until must be timezone-aware")
    return parsed


def _matches(
    event: ChangeEvent,
    *,
    dataset: str | None,
    since: datetime | None,
    until: datetime | None,
    kind: EventKind | None,
) -> bool:
    if dataset is not None and event.dataset != dataset:
        return False
    if kind is not None and event.kind is not kind:
        return False
    if since is not None and event.ts < since:
        return False
    return until is None or event.ts <= until
