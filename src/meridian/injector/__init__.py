"""Seeded fault injection."""

from meridian.injector.cli import main
from meridian.injector.faults import (
    INJECTORS,
    inject_distribution_shift,
    inject_duplicate_keys,
    inject_late_partition,
    inject_null_explosion,
    inject_schema_drift,
    inject_unit_mismatch,
)

__all__ = [
    "INJECTORS",
    "inject_distribution_shift",
    "inject_duplicate_keys",
    "inject_late_partition",
    "inject_null_explosion",
    "inject_schema_drift",
    "inject_unit_mismatch",
    "main",
]
