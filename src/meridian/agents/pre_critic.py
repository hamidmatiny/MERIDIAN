"""Deterministic checks that run before a critic call.

A syntax or sandbox-gate failure goes back to Fix. It does not spend critic
tokens and it does not count as a critic round. A passing patch is dry-run on
a seeded sample: every failing row, up to 150, plus random clean rows to 200.
A repair or quarantine that fixes none of those rows is a NO_OP and returns
to Fix without a critic round. A repair that changes rows outside the failing
set is OVER_BROAD and returns the same way.
"""

import ast
import random
from collections import Counter

import pandas as pd
from pydantic import ValidationError

from meridian.agents.patch_text import source_from_diff
from meridian.agents.verifier import _load, _partition
from meridian.contracts.dry_run import DryRunReport
from meridian.contracts.incident_case import MAX_FIX_ATTEMPTS
from meridian.contracts.reference_profile import ReferenceProfile
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.state import MeridianState
from meridian.quality.evaluate import evaluate, failure_positions
from meridian.quality.psi import population_stability_index
from meridian.sandbox.gate import check_source

_SAMPLE = 200
_FAILING_CAP = 150


def run_pre_critic(
    state: MeridianState,
    *,
    num_cpus: int,
    timeout_s: float,
    memory_mb: int,
    sample_seed: int = 42,
    psi_min_rows: int = 200,
) -> dict[str, object]:
    """Gate the patch, then either bounce to Fix or attach a dry-run report."""
    if len(state.cases) != 1:
        raise ValueError("a pre-critic branch handles one incident")
    key, case = next(iter(state.cases.items()))
    if case.patch is None:
        raise ValueError("pre-critic runs after a patch")
    source = source_from_diff(case.patch.diff)
    blocked = _static(source, case.patch.regression_test)
    if blocked is not None:
        updated = case.model_copy(
            update={
                "pre_critic_attempt": case.fix_attempts,
                "pre_critic_blocked": True,
                "pre_critic_note": blocked,
                "needs_human": case.needs_human or case.fix_attempts >= MAX_FIX_ATTEMPTS,
            }
        )
        return {"cases": {key: updated}}
    if case.patch.remediation_kind is RemediationKind.ESCALATE:
        skipped = case.model_copy(
            update={
                "pre_critic_attempt": case.fix_attempts,
                "pre_critic_blocked": False,
                "pre_critic_note": "",
                "dry_run": None,
                "needs_human": True,
            }
        )
        return {"cases": {key: skipped}}
    if case.patch.remediation_kind is RemediationKind.RERUN_UPSTREAM:
        skipped = case.model_copy(
            update={
                "pre_critic_attempt": case.fix_attempts,
                "pre_critic_blocked": False,
                "pre_critic_note": "",
                "dry_run": None,
            }
        )
        return {"cases": {key: skipped}}
    report = _dry_run(
        state,
        source,
        num_cpus=num_cpus,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        sample_seed=sample_seed,
        psi_min_rows=psi_min_rows,
    )
    if report.no_op:
        updated = case.model_copy(
            update={
                "pre_critic_attempt": case.fix_attempts,
                "pre_critic_blocked": True,
                "pre_critic_note": "NO_OP: rows_fixed=0",
                "dry_run": report,
                "needs_human": case.needs_human or case.fix_attempts >= MAX_FIX_ATTEMPTS,
            }
        )
        return {"cases": {key: updated}}
    if report.over_broad:
        updated = case.model_copy(
            update={
                "pre_critic_attempt": case.fix_attempts,
                "pre_critic_blocked": True,
                "pre_critic_note": (
                    f"OVER_BROAD: rows_changed_outside={report.rows_changed_outside}"
                ),
                "dry_run": report,
                "needs_human": case.needs_human or case.fix_attempts >= MAX_FIX_ATTEMPTS,
            }
        )
        return {"cases": {key: updated}}
    updated = case.model_copy(
        update={
            "pre_critic_attempt": case.fix_attempts,
            "pre_critic_blocked": False,
            "pre_critic_note": "",
            "dry_run": report,
        }
    )
    return {"cases": {key: updated}}


def _static(source: str, regression_test: str) -> str | None:
    reasons: list[str] = []
    reasons.extend(_parsed(source, "remediation"))
    reasons.extend(_parsed(regression_test, "regression test"))
    if not any(item.startswith("remediation syntax") for item in reasons):
        reasons.extend(check_source(source))
    if not any(item.startswith("regression test syntax") for item in reasons):
        reasons.extend(check_source(regression_test, allow_pandera=True))
    if not reasons:
        return None
    return "; ".join(reasons)


def _parsed(source: str, label: str) -> list[str]:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return [f"{label} syntax: {exc.msg}"]
    return []


def _dry_run(
    state: MeridianState,
    source: str,
    *,
    num_cpus: int,
    timeout_s: float,
    memory_mb: int,
    sample_seed: int,
    psi_min_rows: int,
) -> DryRunReport:
    _key, case = next(iter(state.cases.items()))
    assert case.patch is not None
    ref = _partition(state, case.incident.dataset)
    if ref is None:
        return _empty("partition file is missing")
    loaded = _load(ref)
    if isinstance(loaded, str):
        return _empty(loaded)
    frame, profile = loaded
    sample = _sample(
        frame,
        dataset=case.incident.dataset,
        fault=case.incident.fault_type,
        metadata=ref.metadata,
        profile=profile,
        seed=sample_seed,
    )
    kind = case.patch.remediation_kind
    from meridian.sandbox.execute import execute

    entry = "quarantine" if kind is RemediationKind.QUARANTINE else "repair"
    ran = execute(
        source,
        sample,
        entry=entry,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        num_cpus=num_cpus,
    )
    if ran.kind != "ok" or ran.frame is None:
        return _empty(ran.detail or ran.kind)
    return _measured(
        sample,
        ran.frame,
        case.incident.dataset,
        case.incident.fault_type,
        ref.metadata,
        profile,
        quarantine=kind is RemediationKind.QUARANTINE,
        detail="",
        psi_min_rows=psi_min_rows,
    )


def _sample(
    frame: pd.DataFrame,
    *,
    dataset: str,
    fault: object,
    metadata: object,
    profile: ReferenceProfile,
    seed: int,
) -> pd.DataFrame:
    """Keep every failing row up to 150, then seeded clean rows until 200."""
    from meridian.contracts.fault_type import FaultType
    from meridian.contracts.partition_metadata import PartitionMetadata

    failing: list[int] = []
    if isinstance(fault, FaultType) and isinstance(metadata, PartitionMetadata):
        try:
            failing = sorted(failure_positions(dataset, frame, metadata, profile, fault))
        except (ValueError, KeyError):
            failing = []
    chosen = failing[:_FAILING_CAP]
    blocked = set(failing)
    clean = [index for index in range(len(frame)) if index not in blocked]
    need = min(_SAMPLE - len(chosen), len(clean))
    if need > 0:
        chosen = [*chosen, *random.Random(seed).sample(clean, need)]
    if not chosen:
        return frame.iloc[:0].reset_index(drop=True)
    return frame.iloc[chosen].reset_index(drop=True)


def _measured(
    before: pd.DataFrame,
    after: pd.DataFrame,
    dataset: str,
    fault: object,
    metadata: object,
    profile: ReferenceProfile,
    *,
    quarantine: bool,
    detail: str,
    psi_min_rows: int = 200,
) -> DryRunReport:
    from meridian.contracts.fault_type import FaultType
    from meridian.contracts.partition_metadata import PartitionMetadata

    if not isinstance(fault, FaultType) or not isinstance(metadata, PartitionMetadata):
        return _empty("dry-run inputs were not typed")
    failing = failure_positions(dataset, before, metadata, profile, fault)
    changed = _quarantine_changed(before, after) if quarantine else _positional(before, after)
    outside = changed - failing
    after_failing = failure_positions(dataset, after, metadata, profile, fault)
    if len(before) == len(after):
        still = failing & after_failing
        fixed = failing - after_failing
    elif quarantine:
        fixed = failing & changed
        still = failing - fixed
    else:
        fixed = set(range(max(0, len(failing) - len(after_failing))))
        still = set(range(len(after_failing)))
    contracts = _contracts(dataset, after, metadata, profile, psi_min_rows=psi_min_rows)
    return DryRunReport(
        rows_sampled=len(before),
        clean_rows=max(len(before) - len(failing), 0),
        rows_changed=len(changed),
        rows_changed_outside=len(outside),
        over_broad=bool(outside),
        contracts_after=contracts,
        rows_fixed=len(fixed),
        rows_still_failing=len(still),
        no_op=bool(failing) and len(fixed) == 0,
        psi_after=_psi(after, profile, min_rows=psi_min_rows),
        detail=detail,
    )


def changed_positions(before: pd.DataFrame, after: pd.DataFrame, *, quarantine: bool) -> set[int]:
    """Positions the remediation changed. Quarantine matches rows instead of shifting them."""
    if quarantine:
        return _quarantine_changed(before, after)
    return _positional(before, after)


def _positional(before: pd.DataFrame, after: pd.DataFrame) -> set[int]:
    changed: set[int] = set()
    columns = [column for column in before.columns if column in after.columns]
    width = min(len(before), len(after))
    for index in range(width):
        if not _same(before.iloc[index], after.iloc[index], columns):
            changed.add(index)
    if len(before) != len(after):
        changed.update(range(width, max(len(before), len(after))))
    return changed


def _quarantine_changed(before: pd.DataFrame, after: pd.DataFrame) -> set[int]:
    columns = [column for column in before.columns if column in after.columns]
    remaining: Counter[tuple[str, ...]] = Counter(
        _key(after.iloc[index], columns) for index in range(len(after))
    )
    dropped: set[int] = set()
    for index in range(len(before)):
        key = _key(before.iloc[index], columns)
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            dropped.add(index)
    return dropped


def _same(left: pd.Series, right: pd.Series, columns: list[str]) -> bool:
    return all(_cell(left[column]) == _cell(right[column]) for column in columns)


def _key(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(_cell(row[column]) for column in columns)


def _cell(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _contracts(
    dataset: str,
    frame: pd.DataFrame,
    metadata: object,
    profile: ReferenceProfile,
    *,
    psi_min_rows: int = 200,
) -> int:
    from meridian.contracts.partition_metadata import PartitionMetadata

    if not isinstance(metadata, PartitionMetadata):
        return 0
    try:
        return len(
            evaluate(
                dataset,
                frame.reset_index(drop=True),
                metadata,
                profile,
                psi_min_rows=psi_min_rows,
            )
        )
    except (ValueError, ValidationError):
        return 0


def _psi(frame: pd.DataFrame, profile: ReferenceProfile, *, min_rows: int = 0) -> str:
    if len(frame) < min_rows:
        return "skipped_insufficient_rows"
    over: list[str] = []
    for column in profile.columns:
        if column.column not in frame.columns:
            continue
        try:
            values = frame[column.column].to_numpy(dtype=float)
        except (TypeError, ValueError):
            continue
        score = population_stability_index(values, column)
        if score >= profile.psi_threshold:
            over.append(column.column)
    if not over:
        return "under threshold"
    return "over threshold: " + ", ".join(over)


def _empty(detail: str) -> DryRunReport:
    return DryRunReport(
        rows_sampled=0,
        rows_changed=0,
        rows_changed_outside=0,
        over_broad=False,
        contracts_after=0,
        psi_after="",
        detail=detail,
    )
