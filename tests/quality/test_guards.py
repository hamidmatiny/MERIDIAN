"""Rejected profiles, lineage files, and evaluation guards."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.reference_profile import ColumnProfile, ReferenceProfile
from meridian.pipeline import lineage as lineage_module
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE, generate_dataset
from meridian.quality.contracts import model_for
from meridian.quality.evaluate import evaluate
from meridian.quality.freshness import on_time_arrival
from meridian.quality.reference import reference_profile


def test_column_profile_rejects_bad_bins() -> None:
    with pytest.raises(ValidationError, match="one entry per bin"):
        ColumnProfile(column="speed_mps", bin_edges=[0.0, 1.0, 2.0], expected_proportions=[1.0])
    with pytest.raises(ValidationError, match="strictly increasing"):
        ColumnProfile(column="speed_mps", bin_edges=[1.0, 1.0], expected_proportions=[1.0])
    with pytest.raises(ValidationError, match="must be positive"):
        ColumnProfile(column="speed_mps", bin_edges=[0.0, 1.0], expected_proportions=[0.0])


def test_unknown_dataset_has_no_model() -> None:
    with pytest.raises(KeyError):
        model_for("not_a_dataset")


def test_evaluate_requires_matching_names() -> None:
    frame = generate_dataset("fleet_daily_kpi", seed=1, rows=5)
    metadata = PartitionMetadata(
        dataset="trip_summary",
        partition_date=DEFAULT_PARTITION_DATE,
        arrived_at=on_time_arrival(DEFAULT_PARTITION_DATE),
    )
    with pytest.raises(ValueError, match=r"metadata\.dataset"):
        evaluate("fleet_daily_kpi", frame, metadata, reference_profile("fleet_daily_kpi"))
    profile = ReferenceProfile(
        dataset="trip_summary",
        psi_threshold=0.2,
        columns=reference_profile("fleet_daily_kpi").columns,
    )
    on_time = PartitionMetadata(
        dataset="fleet_daily_kpi",
        partition_date=DEFAULT_PARTITION_DATE,
        arrived_at=on_time_arrival(DEFAULT_PARTITION_DATE),
    )
    with pytest.raises(ValueError, match=r"reference_profile\.dataset"):
        evaluate("fleet_daily_kpi", frame, on_time, profile)


def test_evaluate_skips_psi_when_the_column_was_removed() -> None:
    frame = generate_dataset("vehicle_telemetry", seed=2, rows=30).drop(columns=["speed_mps"])
    metadata = PartitionMetadata(
        dataset="vehicle_telemetry",
        partition_date=DEFAULT_PARTITION_DATE,
        arrived_at=on_time_arrival(DEFAULT_PARTITION_DATE),
    )
    incidents = evaluate(
        "vehicle_telemetry",
        frame,
        metadata,
        reference_profile("vehicle_telemetry"),
        detected_at=datetime(2026, 10, 20, tzinfo=UTC),
    )
    assert incidents
    assert all(item.fault_type.value == "SCHEMA_DRIFT" for item in incidents)


def test_lineage_file_must_be_well_formed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "lineage.yaml"
    monkeypatch.setattr(lineage_module, "_LINEAGE_PATH", path)
    path.write_text("note: none\n", encoding="utf-8")
    with pytest.raises(ValueError, match="edges list"):
        lineage_module.downstream_of("vehicle_telemetry")
    path.write_text("edges:\n  - just-a-string\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        lineage_module.downstream_of("vehicle_telemetry")
    path.write_text("edges:\n  - upstream: 1\n    downstream: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="string upstream"):
        lineage_module.downstream_of("vehicle_telemetry")
    path.write_text(
        "edges:\n  - upstream: missing\n    downstream: vehicle_telemetry\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown dataset"):
        lineage_module.downstream_of("vehicle_telemetry")


def test_lineage_walk_skips_a_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "lineage.yaml"
    path.write_text(
        """
edges:
  - {upstream: vehicle_telemetry, downstream: trip_summary}
  - {upstream: trip_summary, downstream: vehicle_telemetry}
  - {upstream: trip_summary, downstream: fleet_daily_kpi}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(lineage_module, "_LINEAGE_PATH", path)
    assert lineage_module.downstream_of("vehicle_telemetry") == [
        "trip_summary",
        "fleet_daily_kpi",
    ]
