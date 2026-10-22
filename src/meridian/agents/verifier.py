"""Deterministic verifier. It replays the partition and does not read a label."""

import time
from pathlib import Path
from uuid import UUID

import pandas as pd
from pydantic import ValidationError

from meridian.agents.patch_text import source_from_diff
from meridian.contracts.dry_run import DryRunReport
from meridian.contracts.partition_ref import PartitionRef
from meridian.contracts.patch import Patch
from meridian.contracts.reference_profile import ReferenceProfile
from meridian.contracts.remediation import RemediationKind, VerificationCheck
from meridian.contracts.state import MeridianState
from meridian.contracts.verification import VerificationResult
from meridian.graph.limits import record_verification
from meridian.pipeline.generate import generate_dataset
from meridian.pipeline.parquet import read_parquet
from meridian.quality.evaluate import evaluate
from meridian.quality.freshness import on_time_arrival
from meridian.quality.psi import population_stability_index
from meridian.sandbox.execute import execute

_CHECK = """\
def check(df):
    return None
"""


def run_verifier(
    state: MeridianState,
    *,
    num_cpus: int,
    timeout_s: float,
    memory_mb: int,
    max_row_loss: float,
    max_quarantine: float,
    max_outside_change: float = 0.0,
) -> dict[str, object]:
    """Replay the current patch. Escalation and a failure at the cap need a human."""
    if len(state.cases) != 1:
        raise ValueError("a verifier branch handles one incident")
    key, case = next(iter(state.cases.items()))
    if case.patch is None:
        raise ValueError("verifier runs after a patch")
    started = time.perf_counter()
    kind = case.patch.remediation_kind
    if kind is RemediationKind.ESCALATE:
        result = _result(
            case.patch.id,
            kind,
            [VerificationCheck(name="escalated", passed=False, detail="No automated fix.")],
            0,
            0,
            started,
        )
        updated = record_verification(case, result, escalate=True)
        return {"cases": {key: updated}}
    ref = _partition(state, case.incident.dataset)
    if ref is None:
        result = _result(
            case.patch.id,
            kind,
            [
                VerificationCheck(
                    name="partition",
                    passed=False,
                    detail="Partition file is missing.",
                )
            ],
            0,
            0,
            started,
        )
        updated = record_verification(case, result, escalate=False)
        return {"cases": {key: updated}}
    if kind is RemediationKind.RERUN_UPSTREAM:
        result = _rerun(case.patch.id, ref, started)
    elif kind is RemediationKind.QUARANTINE:
        result = _quarantine(
            case.patch,
            ref,
            timeout_s=timeout_s,
            memory_mb=memory_mb,
            num_cpus=num_cpus,
            max_quarantine=max_quarantine,
            started=started,
        )
    else:
        result = _repair(
            case.patch,
            ref,
            timeout_s=timeout_s,
            memory_mb=memory_mb,
            num_cpus=num_cpus,
            max_row_loss=max_row_loss,
            dry_run=case.dry_run,
            max_outside_change=max_outside_change,
            started=started,
        )
    updated = record_verification(case, result, escalate=False)
    return {"cases": {key: updated}}


def _repair(
    patch: Patch,
    ref: PartitionRef,
    *,
    timeout_s: float,
    memory_mb: int,
    num_cpus: int,
    max_row_loss: float,
    dry_run: DryRunReport | None,
    max_outside_change: float,
    started: float,
) -> VerificationResult:
    loaded = _load(ref)
    if isinstance(loaded, str):
        return _result(
            patch.id,
            patch.remediation_kind,
            [VerificationCheck(name="partition", passed=False, detail=loaded)],
            0,
            0,
            started,
        )
    frame, profile = loaded
    source = source_from_diff(patch.diff)
    ran = execute(
        source,
        frame,
        entry="repair",
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        num_cpus=num_cpus,
    )
    if ran.kind != "ok" or ran.frame is None:
        return _result(
            patch.id,
            patch.remediation_kind,
            [_sandbox_check(ran.kind, ran.detail)],
            0,
            0,
            started,
        )
    repaired = ran.frame
    contracts, remaining = _contracts(ref, repaired, profile)
    checks = [
        contracts,
        _regression(patch.regression_test, repaired, timeout_s, memory_mb, num_cpus),
        _row_loss(len(frame), len(repaired), max_row_loss),
        _psi(repaired, profile),
        _outside_rows(dry_run, max_outside_change),
    ]
    return _result(patch.id, patch.remediation_kind, checks, remaining, len(repaired), started)


def _quarantine(
    patch: Patch,
    ref: PartitionRef,
    *,
    timeout_s: float,
    memory_mb: int,
    num_cpus: int,
    max_quarantine: float,
    started: float,
) -> VerificationResult:
    loaded = _load(ref)
    if isinstance(loaded, str):
        return _result(
            patch.id,
            patch.remediation_kind,
            [VerificationCheck(name="partition", passed=False, detail=loaded)],
            0,
            0,
            started,
        )
    frame, profile = loaded
    source = source_from_diff(patch.diff)
    ran = execute(
        source,
        frame,
        entry="quarantine",
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        num_cpus=num_cpus,
    )
    if ran.kind != "ok" or ran.frame is None:
        return _result(
            patch.id,
            patch.remediation_kind,
            [_sandbox_check(ran.kind, ran.detail)],
            0,
            0,
            started,
        )
    kept = ran.frame
    dropped = 0 if ran.rows_in == 0 else (ran.rows_in - len(kept)) / ran.rows_in
    fraction = VerificationCheck(
        name="quarantine_fraction",
        passed=dropped <= max_quarantine,
        detail=f"quarantined {dropped:.3f}; cap {max_quarantine:.3f}",
    )
    contracts, remaining = _contracts(ref, kept, profile)
    checks = [
        contracts,
        _regression(patch.regression_test, kept, timeout_s, memory_mb, num_cpus),
        fraction,
    ]
    return _result(patch.id, patch.remediation_kind, checks, remaining, len(kept), started)


def _rerun(patch_id: UUID, ref: PartitionRef, started: float) -> VerificationResult:
    if ref.generation_seed is None or ref.row_count is None:
        checks = [
            VerificationCheck(
                name="rerun_inputs",
                passed=False,
                detail="The partition has no generation seed.",
            )
        ]
        return _result(patch_id, RemediationKind.RERUN_UPSTREAM, checks, 0, 0, started)
    frame = generate_dataset(
        ref.dataset,
        ref.generation_seed,
        rows=ref.row_count,
        partition_date=ref.partition_date,
    )
    profile = ReferenceProfile.model_validate_json(
        Path(ref.reference_profile_path).read_text(encoding="utf-8")
    )
    metadata = ref.metadata.model_copy(update={"arrived_at": on_time_arrival(ref.partition_date)})
    incidents = evaluate(ref.dataset, frame, metadata, profile)
    checks = [
        VerificationCheck(
            name="rerun_inputs",
            passed=True,
            detail="Regenerated from the pipeline seed.",
        ),
        VerificationCheck(
            name="contracts",
            passed=not incidents,
            detail=f"{len(incidents)} incident(s) after the rerun",
        ),
    ]
    return _result(
        patch_id,
        RemediationKind.RERUN_UPSTREAM,
        checks,
        len(incidents),
        len(frame),
        started,
    )


def _contracts(
    ref: PartitionRef,
    frame: pd.DataFrame,
    profile: ReferenceProfile,
) -> tuple[VerificationCheck, int]:
    incidents = evaluate(ref.dataset, frame, ref.metadata, profile)
    return (
        VerificationCheck(
            name="contracts",
            passed=not incidents,
            detail=f"{len(incidents)} incident(s) remain",
        ),
        len(incidents),
    )


def _regression(
    source: str,
    frame: pd.DataFrame,
    timeout_s: float,
    memory_mb: int,
    num_cpus: int,
) -> VerificationCheck:
    ran = execute(
        source or _CHECK,
        frame,
        entry="check",
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        num_cpus=num_cpus,
        allow_pandera=True,
    )
    return VerificationCheck(
        name="regression_test",
        passed=ran.kind == "ok",
        detail=ran.detail or "regression test passed",
    )


def _row_loss(before: int, after: int, cap: float) -> VerificationCheck:
    dropped = 0.0 if before == 0 else max(before - after, 0) / before
    return VerificationCheck(
        name="row_loss",
        passed=dropped <= cap,
        detail=f"dropped {dropped:.3f}; cap {cap:.3f}",
    )


def _outside_rows(dry_run: DryRunReport | None, cap: float) -> VerificationCheck:
    """Fail a repair that rewrites clean rows in the dry-run sample."""
    if dry_run is None:
        return VerificationCheck(
            name="outside_rows",
            passed=False,
            detail="No dry-run sample, so the clean-row change was not measured.",
        )
    changed = dry_run.rows_changed_outside
    clean = dry_run.clean_rows
    ratio = 1.0 if clean == 0 and changed > 0 else (0.0 if clean == 0 else changed / clean)
    return VerificationCheck(
        name="outside_rows",
        passed=ratio <= cap,
        detail=(
            f"rows_changed_outside={changed} clean_rows={clean} ratio={ratio:.3f} cap={cap:.3f}"
        ),
    )


def _psi(frame: pd.DataFrame, profile: ReferenceProfile) -> VerificationCheck:
    over: list[str] = []
    for column in profile.columns:
        if column.column not in frame.columns:
            continue
        score = population_stability_index(frame[column.column].to_numpy(dtype=float), column)
        if score >= profile.psi_threshold:
            over.append(column.column)
    return VerificationCheck(
        name="psi",
        passed=not over,
        detail="under threshold" if not over else "over threshold: " + ", ".join(over),
    )


def _sandbox_check(kind: str, detail: str) -> VerificationCheck:
    name = "static_gate" if kind == "rejected" else kind
    return VerificationCheck(name=name, passed=False, detail=detail or kind)


def _load(ref: PartitionRef) -> tuple[pd.DataFrame, ReferenceProfile] | str:
    try:
        frame = read_parquet(Path(ref.parquet_path))
        profile = ReferenceProfile.model_validate_json(
            Path(ref.reference_profile_path).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        return str(exc)
    return frame, profile


def _partition(state: MeridianState, dataset: str) -> PartitionRef | None:
    for ref in state.partitions:
        if ref.dataset == dataset:
            return ref
    return None


def _result(
    patch_id: UUID,
    kind: RemediationKind,
    checks: list[VerificationCheck],
    contract_failures: int,
    rows: int,
    started: float,
) -> VerificationResult:
    passed = all(item.passed for item in checks)
    failures = 0 if passed else max(contract_failures, 1)
    return VerificationResult(
        patch_id=patch_id,
        remediation_kind=kind,
        checks=checks,
        passed=passed,
        rows_checked=rows,
        failures_remaining=failures,
        duration_ms=max(int((time.perf_counter() - started) * 1000), 0),
    )
