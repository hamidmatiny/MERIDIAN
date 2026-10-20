"""Severity floor and the more-severe ordering."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from meridian.agents.severity_rules import FLOOR_TABLE, floor_for
from meridian.contracts.fault_type import FaultType
from meridian.contracts.severity import SEVERITY_RANK, Severity, more_severe


def test_schema_drift_with_two_downstreams_floors_at_sev2() -> None:
    assert floor_for(FaultType.SCHEMA_DRIFT, 2) is Severity.SEV2
    assert floor_for(FaultType.SCHEMA_DRIFT, 5) is Severity.SEV2


def test_every_fault_has_a_floor_for_each_bucket() -> None:
    for fault in FaultType:
        for bucket in (0, 1, 2):
            assert (fault, bucket) in FLOOR_TABLE
            assert floor_for(fault, bucket) is FLOOR_TABLE[(fault, bucket)]


def test_negative_downstreams_are_rejected() -> None:
    with pytest.raises(ValueError, match="downstreams"):
        floor_for(FaultType.SCHEMA_DRIFT, -1)


def test_missing_floor_row_is_rejected() -> None:
    key = (FaultType.SCHEMA_DRIFT, 0)
    saved = FLOOR_TABLE.pop(key)
    try:
        with pytest.raises(KeyError, match="no severity floor"):
            floor_for(FaultType.SCHEMA_DRIFT, 0)
    finally:
        FLOOR_TABLE[key] = saved


@given(
    llm=st.sampled_from(list(Severity)),
    floor=st.sampled_from(list(Severity)),
)
def test_final_severity_is_the_maximum_rank(llm: Severity, floor: Severity) -> None:
    assert more_severe(llm, floor) is max((llm, floor), key=SEVERITY_RANK.__getitem__)
