"""Seeded change log written beside an injected partition.

Each fault has exactly one causal event. The other events are distractors.
A clean partition gets distractors only.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid5

from meridian.contracts.change_event import ChangeEvent, EventKind
from meridian.contracts.fault_type import FaultType
from meridian.contracts.injected_fault import InjectedFault
from meridian.contracts.namespace import MERIDIAN_NAMESPACE
from meridian.pipeline.catalog import DATASETS

DEFAULT_DISTRACTORS = 8
WINDOW = timedelta(hours=2)
_ANCHOR = datetime(2026, 10, 19, 15, 0, tzinfo=UTC)
_AUTHORS = ("ada", "grace", "linus", "radia")
_DATASETS = tuple(DATASETS)
_KINDS = tuple(EventKind)

CAUSAL_KIND: dict[FaultType, EventKind] = {
    FaultType.SCHEMA_DRIFT: EventKind.SCHEMA_CHANGE,
    FaultType.NULL_EXPLOSION: EventKind.BACKFILL,
    FaultType.UNIT_MISMATCH: EventKind.CONFIG_CHANGE,
    FaultType.DUPLICATE_KEYS: EventKind.DEPLOY,
    FaultType.LATE_PARTITION: EventKind.UPSTREAM_DELAY,
    FaultType.DISTRIBUTION_SHIFT: EventKind.DEPENDENCY_BUMP,
}


def causal_event_id(dataset: str, fault: FaultType, seed: int, intensity: float) -> UUID:
    """Deterministic id of the one event that caused this injection."""
    raw = f"causal|{dataset}|{fault.value}|{seed}|{intensity:.8f}"
    return uuid5(MERIDIAN_NAMESPACE, raw)


def anchor_for(seed: int) -> datetime:
    """UTC time the causal event and the same-window distractor share."""
    return _ANCHOR + timedelta(minutes=seed % 30)


def build_events(
    *,
    dataset: str,
    seed: int,
    fault: FaultType | None,
    columns: list[str],
    intensity: float = 1.0,
    n_distractors: int = DEFAULT_DISTRACTORS,
) -> tuple[list[ChangeEvent], UUID | None]:
    """Build the causal event, when there is a fault, plus ``n_distractors`` others."""
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}")
    if n_distractors < 2:
        raise ValueError("n_distractors must be at least 2")
    anchor = anchor_for(seed)
    causal_id: UUID | None = None
    events: list[ChangeEvent] = []
    if fault is not None:
        causal_id = causal_event_id(dataset, fault, seed, intensity)
        events.append(_causal(causal_id, dataset, fault, columns, anchor))
    for index in range(n_distractors):
        events.append(
            _distractor(
                dataset=dataset,
                seed=seed,
                fault=fault,
                index=index,
                anchor=anchor,
                causal_kind=None if fault is None else CAUSAL_KIND[fault],
            )
        )
    return events, causal_id


def write_events(path: Path, events: list[ChangeEvent]) -> None:
    """Write one JSON object per line."""
    lines = [event.model_dump_json() for event in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_events(path: Path) -> list[ChangeEvent]:
    """Read ``events.jsonl``."""
    events: list[ChangeEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        events.append(ChangeEvent.model_validate_json(line))
    return events


def require_causal_event(fault: InjectedFault, path: Path) -> None:
    """Reject a label whose causal id is missing from ``events.jsonl``."""
    found = [event for event in load_events(path) if event.id == fault.causal_event_id]
    if len(found) != 1:
        raise ValueError("causal_event_id must exist exactly once in events.jsonl")


def _causal(
    event_id: UUID,
    dataset: str,
    fault: FaultType,
    columns: list[str],
    anchor: datetime,
) -> ChangeEvent:
    column = columns[0] if columns else "partition"
    kind = CAUSAL_KIND[fault]
    summary, excerpt = _causal_text(fault, dataset, column)
    return ChangeEvent(
        id=event_id,
        ts=anchor,
        kind=kind,
        dataset=dataset,
        author="pipeline-bot",
        summary=summary,
        diff_excerpt=excerpt,
    )


def _causal_text(fault: FaultType, dataset: str, column: str) -> tuple[str, str]:
    if fault is FaultType.SCHEMA_DRIFT:
        return (
            f"Causal event: renamed {column} on {dataset}",
            f"- {column}\n+ {column}_renamed",
        )
    if fault is FaultType.UNIT_MISMATCH:
        return (
            f"Causal event: switched the unit of {column} on {dataset}",
            f"{column}: source_unit changed",
        )
    if fault is FaultType.LATE_PARTITION:
        return (
            f"Causal event: upstream delay for {dataset}",
            "arrival slipped past the freshness SLA",
        )
    if fault is FaultType.NULL_EXPLOSION:
        return (
            f"Causal event: backfill wrote nulls into {column} on {dataset}",
            f"{column}: null",
        )
    if fault is FaultType.DUPLICATE_KEYS:
        return (
            f"Causal event: deploy wrote duplicate keys on {dataset}",
            f"duplicate {column}",
        )
    return (
        f"Causal event: dependency bump shifted {column} on {dataset}",
        f"{column}: distribution moved",
    )


def _distractor(
    *,
    dataset: str,
    seed: int,
    fault: FaultType | None,
    index: int,
    anchor: datetime,
    causal_kind: EventKind | None,
) -> ChangeEvent:
    fault_name = "clean" if fault is None else fault.value
    event_id = uuid5(MERIDIAN_NAMESPACE, f"distractor|{dataset}|{seed}|{index}|{fault_name}")
    if index == 0:
        chosen = dataset
        moment = anchor - timedelta(days=3)
        kind = _other_kind(causal_kind, EventKind.DEPLOY)
    elif index == 1:
        chosen = _other_dataset(dataset)
        moment = anchor + timedelta(minutes=30)
        kind = _other_kind(causal_kind, EventKind.CONFIG_CHANGE)
    else:
        chosen = _DATASETS[(seed + index) % len(_DATASETS)]
        moment = anchor - timedelta(days=index + 1)
        kind = _KINDS[(seed + index) % len(_KINDS)]
    return ChangeEvent(
        id=event_id,
        ts=moment,
        kind=kind,
        dataset=chosen,
        author=_AUTHORS[index % len(_AUTHORS)],
        summary=f"Distractor: routine {kind.value} on {chosen}",
        diff_excerpt="unrelated change",
    )


def _other_dataset(dataset: str) -> str:
    for name in _DATASETS:
        if name != dataset:
            return name
    return dataset


def _other_kind(causal: EventKind | None, preferred: EventKind) -> EventKind:
    if causal is None or preferred is not causal:
        return preferred
    for kind in _KINDS:
        if kind is not causal:
            return kind
    return preferred
