"""The seeded event log has one causal event and deterministic distractors."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from meridian.contracts.change_event import EventKind
from meridian.contracts.fault_type import FaultType
from meridian.injector.events import (
    WINDOW,
    anchor_for,
    build_events,
    causal_event_id,
    load_events,
    require_causal_event,
    write_events,
)
from meridian.injector.faults import INJECTORS
from meridian.pipeline.generate import generate_dataset

_KIND = {
    FaultType.SCHEMA_DRIFT: EventKind.SCHEMA_CHANGE,
    FaultType.UNIT_MISMATCH: EventKind.CONFIG_CHANGE,
    FaultType.LATE_PARTITION: EventKind.UPSTREAM_DELAY,
    FaultType.NULL_EXPLOSION: EventKind.BACKFILL,
    FaultType.DUPLICATE_KEYS: EventKind.DEPLOY,
    FaultType.DISTRIBUTION_SHIFT: EventKind.DEPENDENCY_BUMP,
}


def test_each_fault_has_exactly_one_causal_event(tmp_path: Path) -> None:
    frame = generate_dataset("vehicle_telemetry", 4, rows=30)
    for fault in FaultType:
        injected, label = INJECTORS[fault](
            frame,
            np.random.default_rng(4),
            0.3,
            dataset="vehicle_telemetry",
            seed=4,
        )
        del injected
        events, causal = build_events(
            dataset="vehicle_telemetry",
            seed=4,
            fault=fault,
            columns=list(label.columns_affected),
            intensity=0.3,
        )
        assert causal == label.causal_event_id
        assert causal == causal_event_id("vehicle_telemetry", fault, 4, 0.3)
        matches = [event for event in events if event.id == causal]
        assert len(matches) == 1
        assert matches[0].summary.startswith("Causal event:")
        assert matches[0].kind is _KIND[fault]
        path = tmp_path / f"{fault.value}.jsonl"
        write_events(path, events)
        require_causal_event(label, path)
        assert load_events(path)[0].id == causal


def test_distractors_are_deterministic_and_include_dataset_and_window() -> None:
    first, causal = build_events(
        dataset="trip_summary",
        seed=9,
        fault=FaultType.UNIT_MISMATCH,
        columns=["avg_speed_mps"],
        intensity=0.1,
    )
    second, again = build_events(
        dataset="trip_summary",
        seed=9,
        fault=FaultType.UNIT_MISMATCH,
        columns=["avg_speed_mps"],
        intensity=0.1,
    )
    assert [event.id for event in first] == [event.id for event in second]
    assert causal == again
    distractors = [event for event in first if event.id != causal]
    assert len(distractors) == 8
    anchor = anchor_for(9)
    assert any(event.dataset == "trip_summary" for event in distractors)
    assert any(abs(event.ts - anchor) <= WINDOW for event in distractors)
    same_dataset = next(event for event in distractors if event.dataset == "trip_summary")
    assert same_dataset.ts == anchor - timedelta(days=3)
    in_window = next(event for event in distractors if abs(event.ts - anchor) <= WINDOW)
    assert in_window.dataset != "trip_summary"
    assert in_window.ts == anchor + timedelta(minutes=30)


def test_clean_partitions_have_distractors_only() -> None:
    events, causal = build_events(
        dataset="fleet_daily_kpi",
        seed=2,
        fault=None,
        columns=[],
    )
    assert causal is None
    assert len(events) == 8
    assert all(not event.summary.startswith("Causal event:") for event in events)
    again, _none = build_events(dataset="fleet_daily_kpi", seed=2, fault=None, columns=[])
    assert [event.model_dump() for event in events] == [event.model_dump() for event in again]


def test_causal_event_must_be_in_the_log(tmp_path: Path) -> None:
    events, _causal = build_events(
        dataset="vehicle_telemetry",
        seed=1,
        fault=FaultType.SCHEMA_DRIFT,
        columns=["speed_mps"],
        intensity=0.3,
    )
    path = tmp_path / "events.jsonl"
    write_events(path, events[1:])
    frame = generate_dataset("vehicle_telemetry", 1, rows=20)
    _injected, label = INJECTORS[FaultType.SCHEMA_DRIFT](
        frame,
        np.random.default_rng(1),
        0.3,
        dataset="vehicle_telemetry",
        seed=1,
    )
    with pytest.raises(ValueError, match="exactly once"):
        require_causal_event(label, path)


def test_event_builder_rejects_bad_inputs() -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        build_events(dataset="missing", seed=1, fault=None, columns=[])
    with pytest.raises(ValueError, match="at least 2"):
        build_events(
            dataset="vehicle_telemetry",
            seed=1,
            fault=None,
            columns=[],
            n_distractors=1,
        )
