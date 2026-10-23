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
from meridian.pipeline.catalog import DATASETS
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
        description=(
            "True when the remediated rows match the clean frame on its primary key. "
            "Identical extra copies of a key count as removed, so the row counts may differ."
        ),
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
    keys = _primary_key(injected_fault.dataset)
    expected = clean_frame
    if kind is not RemediationKind.DATA_REPAIR:
        expected = _kept_clean(clean_frame, corrupted, keys)
    return FixScore(
        kind_appropriate=appropriate,
        repaired_matches_clean=_frames_match(ran.frame, expected, keys),
        rows_lost_pct=lost,
    )


def _primary_key(dataset: str) -> tuple[str, ...]:
    spec = DATASETS.get(dataset)
    if spec is None:
        return ()
    return spec.primary_key


def _kept_clean(
    clean: pd.DataFrame,
    corrupted: pd.DataFrame,
    keys: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Clean rows whose corrupted counterpart was not changed."""
    changed = _changed_rows(clean, corrupted, keys)
    return clean.loc[~changed].reset_index(drop=True)


def _changed_rows(
    clean: pd.DataFrame,
    corrupted: pd.DataFrame,
    keys: tuple[str, ...] = (),
) -> pd.Series:
    """Row flags on ``clean``. Extra copies of a primary key are not a clean row."""
    if keys and all(key in clean.columns and key in corrupted.columns for key in keys):
        return _changed_on_keys(clean, corrupted, keys)
    flags = pd.Series(False, index=clean.index)
    shared_len = min(len(clean), len(corrupted))
    if len(corrupted) < len(clean):
        flags.iloc[shared_len:] = True
    left_frame = clean.iloc[:shared_len]
    right_frame = corrupted.iloc[:shared_len]
    shared = [column for column in left_frame.columns if column in right_frame.columns]
    for column in shared:
        differ = _column_differs(left_frame[column], right_frame[column])
        flags.iloc[:shared_len] = flags.iloc[:shared_len].to_numpy() | differ
    if list(clean.columns) != list(corrupted.columns):
        flags[:] = True
    return flags


def _changed_on_keys(
    clean: pd.DataFrame,
    corrupted: pd.DataFrame,
    keys: tuple[str, ...],
) -> pd.Series:
    """A clean key is changed when its first corrupted copy differs, or the key is gone."""
    flags = pd.Series(True, index=clean.index)
    if list(clean.columns) != list(corrupted.columns):
        return flags
    right = corrupted.drop_duplicates(list(keys), keep="first")
    left = clean.reset_index(drop=True)
    merged = left.merge(
        right,
        on=list(keys),
        how="left",
        suffixes=("", "_corrupted"),
        indicator=True,
    )
    matched = merged["_merge"].to_numpy() == "both"
    differ = np.zeros(len(left), dtype=bool)
    for column in left.columns:
        if column in keys:
            continue
        corrupted_column = f"{column}_corrupted"
        if corrupted_column not in merged.columns:
            differ[:] = True
            break
        differ = differ | _column_differs(merged[column], merged[corrupted_column])
    return pd.Series(~matched | differ, index=clean.index)


def _frames_match(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: tuple[str, ...] = (),
) -> bool:
    """Key-aligned compare. Identical duplicate keys are removed before the value check."""
    if keys:
        return _match_on_keys(left, right, keys)
    order = list(left.columns) if list(left.columns) == list(right.columns) else []
    return _match_aligned(left, right, order)


def _match_on_keys(left: pd.DataFrame, right: pd.DataFrame, keys: tuple[str, ...]) -> bool:
    if any(key not in left.columns or key not in right.columns for key in keys):
        return False
    if list(left.columns) != list(right.columns):
        return False
    left_rows = _collapse_identical(left, keys)
    right_rows = _collapse_identical(right, keys)
    if left_rows is None or right_rows is None:
        return False
    return _match_aligned(left_rows, right_rows, list(keys))


def _collapse_identical(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame | None:
    """One row per key. ``None`` when the same key has two different rows."""
    if frame.empty:
        return frame.reset_index(drop=True)
    unique_rows = frame.drop_duplicates(keep="first")
    if unique_rows.duplicated(list(keys)).any():
        return None
    return unique_rows.drop_duplicates(list(keys), keep="first").reset_index(drop=True)


def _match_aligned(left: pd.DataFrame, right: pd.DataFrame, order: list[str]) -> bool:
    if list(left.columns) != list(right.columns) or not order:
        return False
    if len(left) != len(right):
        return False
    ordered_left = left.sort_values(order, kind="mergesort").reset_index(drop=True)
    ordered_right = right.sort_values(order, kind="mergesort").reset_index(drop=True)
    for column in ordered_left.columns:
        if _column_differs(ordered_left[column], ordered_right[column]).any():
            return False
    return True


def _column_differs(left: pd.Series, right: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        numeric = ~np.isclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            equal_nan=True,
            rtol=1e-5,
            atol=1e-5,
        )
        return np.asarray(numeric, dtype=bool)
    textual = left.astype(str).to_numpy() != right.astype(str).to_numpy()
    return np.asarray(textual, dtype=bool)
