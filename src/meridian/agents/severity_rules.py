"""Deterministic severity floor. The model may raise a severity, never lower it."""

from meridian.contracts.fault_type import FaultType
from meridian.contracts.severity import Severity

# Key is (fault, downstream count). Count 2 means two or more downstream datasets.
# SCHEMA_DRIFT with 2 or more downstreams floors at SEV2.
FLOOR_TABLE: dict[tuple[FaultType, int], Severity] = {
    (FaultType.SCHEMA_DRIFT, 0): Severity.SEV3,
    (FaultType.SCHEMA_DRIFT, 1): Severity.SEV2,
    (FaultType.SCHEMA_DRIFT, 2): Severity.SEV2,
    (FaultType.NULL_EXPLOSION, 0): Severity.SEV3,
    (FaultType.NULL_EXPLOSION, 1): Severity.SEV2,
    (FaultType.NULL_EXPLOSION, 2): Severity.SEV2,
    (FaultType.UNIT_MISMATCH, 0): Severity.SEV3,
    (FaultType.UNIT_MISMATCH, 1): Severity.SEV2,
    (FaultType.UNIT_MISMATCH, 2): Severity.SEV1,
    (FaultType.DUPLICATE_KEYS, 0): Severity.SEV4,
    (FaultType.DUPLICATE_KEYS, 1): Severity.SEV3,
    (FaultType.DUPLICATE_KEYS, 2): Severity.SEV2,
    (FaultType.LATE_PARTITION, 0): Severity.SEV4,
    (FaultType.LATE_PARTITION, 1): Severity.SEV3,
    (FaultType.LATE_PARTITION, 2): Severity.SEV2,
    (FaultType.DISTRIBUTION_SHIFT, 0): Severity.SEV3,
    (FaultType.DISTRIBUTION_SHIFT, 1): Severity.SEV3,
    (FaultType.DISTRIBUTION_SHIFT, 2): Severity.SEV2,
}


def floor_for(fault: FaultType, downstreams: int) -> Severity:
    """Minimum severity for this fault and blast-radius size."""
    if downstreams < 0:
        raise ValueError("downstreams must be >= 0")
    bucket = 2 if downstreams >= 2 else downstreams
    try:
        return FLOOR_TABLE[(fault, bucket)]
    except KeyError as exc:
        raise KeyError(
            f"no severity floor for {fault.value} with {downstreams} downstreams"
        ) from exc
