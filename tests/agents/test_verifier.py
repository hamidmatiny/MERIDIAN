"""Verifier checks that do not need the critic loop."""

from pathlib import Path
from uuid import uuid4

from tests.agents.support import write_partition
from tests.contracts.builders import make_hypothesis

from meridian.agents.patch_text import unified_diff
from meridian.agents.verifier import run_verifier
from meridian.contracts.dry_run import DryRunReport
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.patch import Patch
from meridian.contracts.reference_profile import ReferenceProfile
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.state import MeridianState
from meridian.pipeline.parquet import read_parquet
from meridian.quality.evaluate import evaluate


def _patch(incident_id, hypothesis_id, kind: RemediationKind, source: str) -> Patch:
    path = f"remediations/{incident_id}.py"
    return Patch(
        id=uuid4(),
        incident_id=incident_id,
        hypothesis_id=hypothesis_id,
        remediation_kind=kind,
        diff=unified_diff(path, source),
        regression_test=(
            "def check(df):\n    import pandera as pa\n    pa.DataFrameSchema({}).validate(df)\n"
        ),
        files_touched=[path],
        author_model="test",
    )


def _state(tmp_path: Path, fault: FaultType, kind: RemediationKind, source: str) -> MeridianState:
    ref = write_partition(
        tmp_path,
        dataset="fleet_daily_kpi",
        fault=fault,
        intensity=0.1,
        seed=5,
        rows=400,
        name="part",
    )
    profile = ReferenceProfile.model_validate_json(
        Path(ref.reference_profile_path).read_text(encoding="utf-8")
    )
    incidents = evaluate(
        ref.dataset,
        read_parquet(Path(ref.parquet_path)),
        ref.metadata,
        profile,
    )
    incident = next(item for item in incidents if item.fault_type is fault)
    hypothesis = make_hypothesis(incident_id=incident.id)
    case = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=_patch(incident.id, hypothesis.id, kind, source),
    )
    return MeridianState(run_id=uuid4(), partitions=[ref], cases={str(incident.id): case})


def test_rerun_upstream_regenerates_a_passing_partition(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        FaultType.LATE_PARTITION,
        RemediationKind.RERUN_UPSTREAM,
        "def repair(df):\n    return df\n",
    )
    updated = run_verifier(
        state,
        num_cpus=1,
        timeout_s=20,
        memory_mb=256,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    case = next(iter(updated["cases"].values()))
    assert case.verification is not None
    assert case.verification.passed
    assert case.needs_human is False
    assert {item.name for item in case.verification.checks} == {"rerun_inputs", "contracts"}


def test_rerun_without_a_seed_fails(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        FaultType.LATE_PARTITION,
        RemediationKind.RERUN_UPSTREAM,
        "def repair(df):\n    return df\n",
    )
    ref = state.partitions[0].model_copy(update={"generation_seed": None, "row_count": None})
    bare = state.model_copy(update={"partitions": [ref]})
    updated = run_verifier(
        bare,
        num_cpus=1,
        timeout_s=20,
        memory_mb=256,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    case = next(iter(updated["cases"].values()))
    assert case.verification is not None
    assert case.verification.passed is False
    assert case.verification.checks[0].name == "rerun_inputs"


def test_quarantine_keeps_the_remaining_rows(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        FaultType.NULL_EXPLOSION,
        RemediationKind.QUARANTINE,
        "def quarantine(df):\n    return df.isna().any(axis=1)\n",
    )
    updated = run_verifier(
        state,
        num_cpus=1,
        timeout_s=30,
        memory_mb=256,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    case = next(iter(updated["cases"].values()))
    assert case.verification is not None
    assert case.verification.passed
    names = [item.name for item in case.verification.checks]
    assert names == ["contracts", "regression_test", "quarantine_fraction"]


def test_missing_partition_is_a_failure(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        FaultType.LATE_PARTITION,
        RemediationKind.DATA_REPAIR,
        "def repair(df):\n    return df\n",
    )
    updated = run_verifier(
        state.model_copy(update={"partitions": []}),
        num_cpus=1,
        timeout_s=5,
        memory_mb=128,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    case = next(iter(updated["cases"].values()))
    assert case.verification is not None
    assert case.verification.checks[0].name == "partition"
    assert case.verification.passed is False


def test_the_live_over_broad_trace_fails_outside_rows(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        FaultType.UNIT_MISMATCH,
        RemediationKind.DATA_REPAIR,
        "def repair(df):\n    return df\n",
    )
    key, case = next(iter(state.cases.items()))
    report = DryRunReport(
        rows_sampled=200,
        clean_rows=50,
        rows_changed=200,
        rows_changed_outside=50,
        over_broad=True,
        contracts_after=0,
        rows_fixed=150,
        rows_still_failing=0,
    )
    traced = state.model_copy(update={"cases": {key: case.model_copy(update={"dry_run": report})}})
    updated = run_verifier(
        traced,
        num_cpus=1,
        timeout_s=20,
        memory_mb=256,
        max_row_loss=0.01,
        max_quarantine=0.25,
        max_outside_change=0.0,
    )
    checked = next(iter(updated["cases"].values()))
    assert checked.verification is not None
    outside = next(item for item in checked.verification.checks if item.name == "outside_rows")
    assert outside.passed is False
    assert "rows_changed_outside=50" in outside.detail
    assert "clean_rows=50" in outside.detail
    assert checked.verification.passed is False


def test_verifier_does_not_read_the_injected_fault() -> None:
    from meridian.agents import verifier

    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "INJECTORS" not in source
    assert "InjectedFault" not in source
    assert "clean_frame" not in source
