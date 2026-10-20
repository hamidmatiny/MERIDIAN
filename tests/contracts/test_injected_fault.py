"""InjectedFault delay rules."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from meridian.contracts.fault_type import FaultType
from meridian.contracts.injected_fault import InjectedFault


def _label(**overrides: object) -> InjectedFault:
    payload: dict[str, object] = {
        "id": UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a11"),
        "fault_type": FaultType.NULL_EXPLOSION,
        "dataset": "vehicle_telemetry",
        "columns_affected": ["battery_pct"],
        "rows_affected": 10,
        "intensity": 0.1,
        "seed": 42,
    }
    payload.update(overrides)
    return InjectedFault.model_validate(payload)


def test_late_partition_requires_a_delay() -> None:
    with pytest.raises(ValidationError, match="requires arrival_delay_s"):
        _label(fault_type=FaultType.LATE_PARTITION, columns_affected=[], rows_affected=0)
    late = _label(
        fault_type=FaultType.LATE_PARTITION,
        columns_affected=[],
        rows_affected=0,
        arrival_delay_s=90,
    )
    assert late.arrival_delay_s == 90


def test_other_faults_reject_a_delay() -> None:
    with pytest.raises(ValidationError, match="only allowed for LATE_PARTITION"):
        _label(arrival_delay_s=90)
    assert _label().arrival_delay_s is None
