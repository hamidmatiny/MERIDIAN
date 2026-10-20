"""Ray Sentinel matches a serial evaluation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.agents.support import DETECTED_AT, write_partition

from meridian.agents.ray_runtime import ray_runtime
from meridian.agents.sentinel import _remote_entry, collect_ray, collect_serial
from meridian.contracts.fault_type import FaultType

_CASES = (
    ("vehicle_telemetry", FaultType.NULL_EXPLOSION, 0.1, 1),
    ("vehicle_telemetry", FaultType.SCHEMA_DRIFT, 0.3, 2),
    ("vehicle_telemetry", FaultType.UNIT_MISMATCH, 0.1, 3),
    ("vehicle_telemetry", FaultType.DUPLICATE_KEYS, 0.1, 4),
    ("trip_summary", FaultType.DISTRIBUTION_SHIFT, 0.3, 5),
    ("trip_summary", FaultType.LATE_PARTITION, 0.2, 6),
    ("fleet_daily_kpi", FaultType.NULL_EXPLOSION, 0.2, 7),
    ("fleet_daily_kpi", FaultType.DUPLICATE_KEYS, 0.5, 8),
)


def test_ray_matches_serial_on_eight_mixed_partitions(tmp_path: Path) -> None:
    refs = [
        write_partition(
            tmp_path,
            dataset=dataset,
            fault=fault,
            intensity=intensity,
            seed=seed,
            rows=80,
            name=f"part-{index}",
        )
        for index, (dataset, fault, intensity, seed) in enumerate(_CASES)
    ]
    serial = collect_serial(refs, DETECTED_AT)
    parallel = collect_ray(refs, DETECTED_AT, num_cpus=2)
    assert parallel == serial
    assert {item.fault_type for item in parallel} >= {
        FaultType.NULL_EXPLOSION,
        FaultType.SCHEMA_DRIFT,
        FaultType.UNIT_MISMATCH,
        FaultType.DUPLICATE_KEYS,
        FaultType.DISTRIBUTION_SHIFT,
        FaultType.LATE_PARTITION,
    }


def test_ray_runtime_rejects_zero_cpus() -> None:
    with pytest.raises(ValueError, match="num_cpus"), ray_runtime(0):
        pass


def test_empty_partition_list_skips_ray() -> None:
    assert collect_ray([], datetime.now(UTC), num_cpus=1) == []


def test_remote_entry_matches_the_in_process_evaluation(tmp_path: Path) -> None:
    ref = write_partition(
        tmp_path,
        dataset="fleet_daily_kpi",
        fault=FaultType.NULL_EXPLOSION,
        intensity=0.2,
        seed=4,
        rows=40,
        name="remote",
    )
    raw = _remote_entry(ref.model_dump_json(), DETECTED_AT.isoformat())
    assert raw
    assert collect_serial([ref], DETECTED_AT)[0].model_dump_json() == raw[0]
