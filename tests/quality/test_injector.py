"""Injector determinism, the CLI, and the check map."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrorReason

from meridian.contracts.fault_type import FaultType
from meridian.injector.cli import main
from meridian.injector.faults import INJECTORS
from meridian.pipeline.generate import generate_dataset
from meridian.pipeline.parquet import read_parquet, write_parquet
from meridian.quality.mapping import CHECK_TO_FAULT, REASON_TO_FAULT, fault_type_for
from meridian.quality.psi import population_stability_index
from meridian.quality.reference import reference_profile


def test_same_seed_rebuilds_frames_and_labels() -> None:
    for dataset in ("vehicle_telemetry", "trip_summary", "fleet_daily_kpi"):
        clean = generate_dataset(dataset, seed=7, rows=300)
        for fault in FaultType:
            kwargs = {"dataset": dataset, "seed": 7}
            first_frame, first_label = INJECTORS[fault](
                clean, np.random.default_rng(7), 0.3, **kwargs
            )
            second_frame, second_label = INJECTORS[fault](
                clean, np.random.default_rng(7), 0.3, **kwargs
            )
            pd.testing.assert_frame_equal(first_frame, second_frame)
            assert first_label == second_label


def test_parquet_bytes_match_for_the_same_seed(tmp_path: Path) -> None:
    frame = generate_dataset("vehicle_telemetry", seed=42, rows=100)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    left, _label_a = INJECTORS[FaultType.DISTRIBUTION_SHIFT](
        frame, rng_a, 0.3, dataset="vehicle_telemetry", seed=42
    )
    right, _label_b = INJECTORS[FaultType.DISTRIBUTION_SHIFT](
        frame, rng_b, 0.3, dataset="vehicle_telemetry", seed=42
    )
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    write_parquet(left, first)
    write_parquet(right, second)
    assert first.read_bytes() == second.read_bytes()
    pd.testing.assert_frame_equal(read_parquet(first), left)


def test_cli_writes_parquet_and_label(tmp_path: Path) -> None:
    code = main(
        [
            "--dataset",
            "vehicle_telemetry",
            "--fault",
            "NULL_EXPLOSION",
            "--intensity",
            "0.1",
            "--seed",
            "42",
            "--rows",
            "50",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert (tmp_path / "vehicle_telemetry.parquet").is_file()
    label = (tmp_path / "injected_fault.json").read_text(encoding="utf-8")
    assert "NULL_EXPLOSION" in label


def test_module_entrypoint_writes_the_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "meridian.injector",
            "--dataset",
            "fleet_daily_kpi",
            "--fault",
            "LATE_PARTITION",
            "--intensity",
            "0.2",
            "--seed",
            "3",
            "--rows",
            "20",
            "--out",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("meridian.injector", run_name="__main__")
    assert caught.value.code == 0
    assert (tmp_path / "injected_fault.json").is_file()


def test_cli_rejects_an_unknown_fault() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--dataset",
                "vehicle_telemetry",
                "--fault",
                "NOT_A_FAULT",
                "--intensity",
                "0.1",
                "--seed",
                "1",
                "--out",
                "data",
            ]
        )


def test_intensity_must_be_inside_the_open_unit_interval() -> None:
    frame = generate_dataset("fleet_daily_kpi", seed=1, rows=10)
    for bad in (0.0, 1.5, True, "nope", float("nan")):
        with pytest.raises(ValueError, match="intensity"):
            INJECTORS[FaultType.NULL_EXPLOSION](
                frame,
                np.random.default_rng(1),
                bad,  # type: ignore[arg-type]
                dataset="fleet_daily_kpi",
                seed=1,
            )
    with pytest.raises(TypeError, match="numpy Generator"):
        INJECTORS[FaultType.NULL_EXPLOSION](
            frame,
            None,  # type: ignore[arg-type]
            0.1,
            dataset="fleet_daily_kpi",
            seed=1,
        )


def test_mapping_covers_every_fault_and_rejects_unknown_checks() -> None:
    assert set(CHECK_TO_FAULT.values()) == set(FaultType)
    assert fault_type_for("not_nullable", None) is FaultType.NULL_EXPLOSION
    assert fault_type_for("less_than_or_equal_to", SchemaErrorReason.CHECK_ERROR) is (
        FaultType.SCHEMA_DRIFT
    )
    assert fault_type_for(None, SchemaErrorReason.DUPLICATES) is FaultType.DUPLICATE_KEYS
    with pytest.raises(KeyError):
        fault_type_for("not_in_the_table", None)
    assert SchemaErrorReason.WRONG_DATATYPE in REASON_TO_FAULT


def test_psi_is_low_on_the_generating_distribution_and_high_after_a_warp() -> None:
    profile = reference_profile("vehicle_telemetry").columns[0]
    values = np.linspace(profile.bin_edges[0], profile.bin_edges[-1], num=2_000, endpoint=False)
    assert population_stability_index(values, profile) < 0.2
    warped = profile.bin_edges[0] + (np.linspace(0, 1, 2_000, endpoint=False) ** 2) * (
        profile.bin_edges[-1] - profile.bin_edges[0]
    )
    assert population_stability_index(warped, profile) > 0.2
    assert population_stability_index(np.array([]), profile) == np.inf
    assert population_stability_index(np.array([1e9, -1e9]), profile) == np.inf
