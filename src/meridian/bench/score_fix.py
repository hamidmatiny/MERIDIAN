"""Score a remediation against the clean frame. The verifier never calls this."""

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from meridian.agents.patch_text import source_from_diff
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.injected_fault import InjectedFault
from meridian.contracts.remediation import RemediationKind
from meridian.injector.faults import INJECTORS
from meridian.sandbox.execute import execute

_APPROPRIATE: dict[FaultType, frozenset[RemediationKind]] = {
    FaultType.NULL_EXPLOSION: frozenset(
        {RemediationKind.QUARANTINE, RemediationKind.ESCALATE, RemediationKind.RERUN_UPSTREAM}
    ),
    FaultType.LATE_PARTITION: frozenset({RemediationKind.RERUN_UPSTREAM, RemediationKind.ESCALATE}),
    FaultType.UNIT_MISMATCH: frozenset({RemediationKind.DATA_REPAIR}),
    FaultType.SCHEMA_DRIFT: frozenset({RemediationKind.DATA_REPAIR, RemediationKind.ESCALATE}),
    FaultType.DISTRIBUTION_SHIFT: frozenset(
        {RemediationKind.DATA_REPAIR, RemediationKind.ESCALATE}
    ),
    FaultType.DUPLICATE_KEYS: frozenset({RemediationKind.DATA_REPAIR, RemediationKind.QUARANTINE}),
}


class FixScore(BaseModel):
    """Three checks a later benchmark can average. This module publishes no accuracy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind_appropriate: bool = Field(description="True when the kind is allowed for this fault.")
    repaired_matches_clean: bool = Field(
        description="True when the remediated rows match the clean frame within tolerance.",
    )
    rows_lost_pct: float = Field(
        description="Rows removed, in percentage points. 1.0 means one percent of the rows.",
    )


def score(
    case: IncidentCase,
    clean_frame: pd.DataFrame,
    injected_fault: InjectedFault,
) -> FixScore:
    """Compare the case's patch to the clean frame. The clean frame is bench-only."""
    patch = case.patch
    if patch is None:
        return FixScore(kind_appropriate=False, repaired_matches_clean=False, rows_lost_pct=0.0)
    kind = patch.remediation_kind
    appropriate = kind in _APPROPRIATE[injected_fault.fault_type]
    if kind is RemediationKind.ESCALATE:
        return FixScore(
            kind_appropriate=appropriate,
            repaired_matches_clean=False,
            rows_lost_pct=0.0,
        )
    if kind is RemediationKind.RERUN_UPSTREAM:
        return FixScore(
            kind_appropriate=appropriate,
            repaired_matches_clean=True,
            rows_lost_pct=0.0,
        )
    corrupted, _label = INJECTORS[injected_fault.fault_type](
        clean_frame,
        np.random.default_rng(injected_fault.seed),
        injected_fault.intensity,
        dataset=injected_fault.dataset,
        seed=injected_fault.seed,
    )
    entry = "quarantine" if kind is RemediationKind.QUARANTINE else "repair"
    ran = execute(
        source_from_diff(patch.diff),
        corrupted,
        entry=entry,
        timeout_s=20,
        memory_mb=512,
        num_cpus=1,
    )
    if ran.kind != "ok" or ran.frame is None:
        return FixScore(
            kind_appropriate=appropriate,
            repaired_matches_clean=False,
            rows_lost_pct=0.0,
        )
    lost = 0.0 if ran.rows_in == 0 else max(ran.rows_in - len(ran.frame), 0) / ran.rows_in * 100
    expected = clean_frame
    if kind is not RemediationKind.DATA_REPAIR:
        expected = _kept_clean(clean_frame, corrupted)
    return FixScore(
        kind_appropriate=appropriate,
        repaired_matches_clean=_frames_match(ran.frame, expected),
        rows_lost_pct=lost,
    )


def _kept_clean(clean: pd.DataFrame, corrupted: pd.DataFrame) -> pd.DataFrame:
    """Clean rows whose corrupted counterpart was not changed."""
    changed = _changed_rows(clean, corrupted)
    return clean.loc[~changed].reset_index(drop=True)


def _changed_rows(clean: pd.DataFrame, corrupted: pd.DataFrame) -> pd.Series:
    flags = pd.Series(False, index=clean.index)
    shared = [column for column in clean.columns if column in corrupted.columns]
    for column in shared:
        left = clean[column]
        right = corrupted[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            differ = ~np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-5,
                atol=1e-5,
            )
        else:
            differ = left.astype(str).to_numpy() != right.astype(str).to_numpy()
        flags = flags | pd.Series(differ, index=clean.index)
    if list(clean.columns) != list(corrupted.columns):
        flags[:] = True
    return flags


def _frames_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    ordered_left = left.sort_values(list(left.columns), kind="mergesort").reset_index(drop=True)
    ordered_right = right.sort_values(list(right.columns), kind="mergesort").reset_index(drop=True)
    for column in ordered_left.columns:
        a = ordered_left[column]
        b = ordered_right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(
                a.to_numpy(dtype=float),
                b.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-5,
                atol=1e-5,
            ):
                return False
        elif not a.astype(str).equals(b.astype(str)):
            return False
    return True
