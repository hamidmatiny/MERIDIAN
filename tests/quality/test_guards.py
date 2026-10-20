"""Rejected profiles, lineage files, and evaluation guards."""

import ast
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.reference_profile import ColumnProfile, ReferenceProfile
from meridian.pipeline import lineage as lineage_module
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE, generate_dataset
from meridian.quality.contracts import model_for
from meridian.quality.evaluate import evaluate
from meridian.quality.freshness import on_time_arrival
from meridian.quality.reference import fit_reference_profile


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
    frame = generate_dataset("fleet_daily_kpi", seed=1, rows=50)
    fitted = fit_reference_profile(generate_dataset("fleet_daily_kpi", seed=9_001, rows=50))
    metadata = PartitionMetadata(
        dataset="trip_summary",
        partition_date=DEFAULT_PARTITION_DATE,
        arrived_at=on_time_arrival(DEFAULT_PARTITION_DATE),
    )
    with pytest.raises(ValueError, match=r"metadata\.dataset"):
        evaluate("fleet_daily_kpi", frame, metadata, fitted)
    profile = fitted.model_copy(update={"dataset": "trip_summary"})
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
    profile = ReferenceProfile(
        dataset="vehicle_telemetry",
        psi_threshold=0.2,
        columns=[
            ColumnProfile(column="speed_mps", bin_edges=[0.0, 1.0], expected_proportions=[1.0])
        ],
    )
    incidents = evaluate(
        "vehicle_telemetry",
        frame,
        metadata,
        profile,
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


def test_quality_does_not_import_the_generator() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "meridian" / "quality"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            imported = any(
                module == "meridian.pipeline.generate" or module.endswith(".generate")
                for module in modules
            )
            if imported:
                offenders.append(str(path))
            if any(name == "generate" for name in names) and any(
                "pipeline" in module for module in modules
            ):
                offenders.append(str(path))
    assert offenders == []


def test_fit_reference_profile_uses_the_frame_quantiles() -> None:
    values = np.linspace(0.0, 100.0, 1001)
    frame = pd.DataFrame({"speed_mps": values})
    frame.attrs["dataset"] = "vehicle_telemetry"
    profile = fit_reference_profile(frame)
    assert profile.dataset == "vehicle_telemetry"
    assert [column.column for column in profile.columns] == ["speed_mps"]
    edges = profile.columns[0].bin_edges
    assert edges[0] == 0.0
    assert edges[-1] == 100.0
    assert len(edges) == 11
    bare = pd.DataFrame({"speed_mps": values})
    with pytest.raises(ValueError, match="dataset name"):
        fit_reference_profile(bare)


def test_fit_rejects_a_frame_with_no_usable_number() -> None:
    text = pd.DataFrame({"vehicle_id": ["a", "b"]})
    text.attrs["dataset"] = "vehicle_telemetry"
    with pytest.raises(ValueError, match="no numeric"):
        fit_reference_profile(text)
    single = pd.DataFrame({"speed_mps": [1.0, np.nan]})
    single.attrs["dataset"] = "vehicle_telemetry"
    with pytest.raises(ValueError, match="no numeric"):
        fit_reference_profile(single)
    flat = pd.DataFrame({"speed_mps": [3.0, 3.0, 3.0]})
    flat.attrs["dataset"] = "vehicle_telemetry"
    with pytest.raises(ValueError, match="no numeric"):
        fit_reference_profile(flat)
