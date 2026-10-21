"""Syntax and gate failures never spend a critic round."""

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from tests.agents.support import write_partition
from tests.contracts.builders import make_hypothesis, make_incident, make_patch

from meridian.agents.critic import run_critic
from meridian.agents.patch_text import unified_diff
from meridian.agents.pre_critic import (
    _contracts,
    _measured,
    _psi,
    changed_positions,
    run_pre_critic,
)
from meridian.contracts.dry_run import DryRunReport
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.reference_profile import ReferenceProfile
from meridian.contracts.remediation import ObjectionKind, RemediationKind
from meridian.contracts.state import MeridianState
from meridian.contracts.verdict import Verdict
from meridian.graph.limits import record_verdict, rejection_rate
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE
from meridian.quality.evaluate import failure_positions
from meridian.quality.freshness import late_arrival


def test_syntax_failure_returns_to_fix_without_a_critic_round() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id).model_copy(
        update={
            "diff": unified_diff("remediations/x.py", "def repair(\n"),
            "regression_test": "def check(df):\n    return None\n",
        }
    )
    case = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        fix_attempts=1,
    )
    update = run_pre_critic(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        num_cpus=1,
        timeout_s=1,
        memory_mb=64,
    )
    updated = update["cases"][str(incident.id)]
    assert updated.pre_critic_blocked is True
    assert updated.critic_rounds == 0
    assert updated.pre_critic_note.startswith("remediation syntax")
    assert "llm_calls" not in update


def test_import_outside_the_gate_is_not_a_critic_round() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    source = "import os\n\ndef repair(df):\n    return df\n"
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id).model_copy(
        update={
            "diff": unified_diff("remediations/x.py", source),
            "regression_test": "def check(df):\n    return None\n",
        }
    )
    case = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        fix_attempts=1,
    )
    update = run_pre_critic(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        num_cpus=1,
        timeout_s=1,
        memory_mb=64,
    )
    updated = update["cases"][str(incident.id)]
    assert updated.pre_critic_blocked is True
    assert "import os" in updated.pre_critic_note
    assert updated.critic_rounds == 0


def test_escalate_and_rerun_leave_dry_run_unset() -> None:
    source = "def repair(df):\n    return df\n"
    for kind in (RemediationKind.ESCALATE, RemediationKind.RERUN_UPSTREAM):
        incident = make_incident()
        hypothesis = make_hypothesis(incident_id=incident.id)
        patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id).model_copy(
            update={
                "remediation_kind": kind,
                "diff": unified_diff("remediations/x.py", source),
                "regression_test": "def check(df):\n    return None\n",
            }
        )
        case = IncidentCase(
            incident=incident,
            hypothesis=hypothesis,
            patch=patch,
            fix_attempts=1,
        )
        update = run_pre_critic(
            MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
            num_cpus=1,
            timeout_s=1,
            memory_mb=64,
        )
        updated = update["cases"][str(incident.id)]
        assert updated.pre_critic_blocked is False
        assert updated.dry_run is None
        assert updated.critic_rounds == 0


def test_over_broad_is_a_row_flag() -> None:
    before = pd.DataFrame({"speed_mps": [10.0, 80.0, 12.0]})
    after = pd.DataFrame({"speed_mps": [10.0 / 3.6, 80.0 / 3.6, 12.0 / 3.6]})
    changed = changed_positions(before, after, quarantine=False)
    assert changed == {0, 1, 2}


def test_over_broad_objection_requires_the_flag() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    case = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        dry_run=DryRunReport(
            rows_sampled=3,
            rows_changed=1,
            rows_changed_outside=0,
            over_broad=False,
            contracts_after=0,
            psi_after="under threshold",
        ),
    )

    class _Reject:
        def with_structured_output(self, schema: object, **kwargs: object) -> "_Reject":
            del schema, kwargs
            return self

        def invoke(self, messages: object, **kwargs: object) -> dict[str, object]:
            del messages, kwargs
            return {
                "raw": None,
                "parsed": {
                    "approved": False,
                    "objections": [
                        {"kind": "OVER_BROAD_CHANGE", "message": "rows_changed=1 outside=0"},
                    ],
                },
                "parsing_error": None,
            }

    update = run_critic(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        model_name="gemma4:e4b",
        chat=_Reject(),
    )
    verdict = update["cases"][str(incident.id)].verdicts[-1]
    assert verdict.critic_degraded is False
    assert [item.kind for item in verdict.objections] == [ObjectionKind.OTHER]


def test_two_degraded_reviews_need_a_human() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    case = IncidentCase(incident=incident, hypothesis=hypothesis, patch=patch)
    for round_number in (1, 2):
        case = record_verdict(
            case,
            Verdict(
                patch_id=patch.id,
                reviewer_model="gemma4:e4b",
                approved=False,
                objections=[],
                critic_degraded=True,
                round=round_number,
            ),
        )
    assert case.needs_human is True
    assert case.critic_rounds == 2
    assert rejection_rate(case.verdicts) == 0.0


def test_a_regression_syntax_error_skips_the_gate() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id).model_copy(
        update={
            "diff": unified_diff("remediations/x.py", "def repair(df):\n    return df\n"),
            "regression_test": "def check(\n",
        }
    )
    case = IncidentCase(incident=incident, hypothesis=hypothesis, patch=patch, fix_attempts=1)
    update = run_pre_critic(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        num_cpus=1,
        timeout_s=1,
        memory_mb=64,
    )
    note = update["cases"][str(incident.id)].pre_critic_note
    assert "regression test syntax" in note


def test_pre_critic_rejects_a_missing_patch() -> None:
    incident = make_incident()
    state = MeridianState(run_id=uuid4(), cases={str(incident.id): IncidentCase(incident=incident)})
    with pytest.raises(ValueError, match="one incident"):
        run_pre_critic(MeridianState(run_id=uuid4()), num_cpus=1, timeout_s=1, memory_mb=64)
    with pytest.raises(ValueError, match="after a patch"):
        run_pre_critic(state, num_cpus=1, timeout_s=1, memory_mb=64)


def test_dry_run_executes_a_repair_and_a_failed_one(tmp_path: Path) -> None:
    ref = write_partition(
        tmp_path,
        dataset="vehicle_telemetry",
        fault=FaultType.UNIT_MISMATCH,
        intensity=0.2,
        seed=1,
        rows=30,
        name="unit",
    )
    incident = make_incident().model_copy(
        update={"dataset": ref.dataset, "fault_type": FaultType.UNIT_MISMATCH}
    )
    hypothesis = make_hypothesis(incident_id=incident.id)
    repair = "def repair(df):\n    out = df.copy()\n    return out\n"
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id).model_copy(
        update={
            "remediation_kind": RemediationKind.DATA_REPAIR,
            "diff": unified_diff("remediations/x.py", repair),
            "regression_test": "def check(df):\n    return None\n",
        }
    )
    case = IncidentCase(incident=incident, hypothesis=hypothesis, patch=patch, fix_attempts=1)
    update = run_pre_critic(
        MeridianState(run_id=uuid4(), partitions=[ref], cases={str(incident.id): case}),
        num_cpus=1,
        timeout_s=30,
        memory_mb=512,
    )
    report = update["cases"][str(incident.id)].dry_run
    assert report is not None
    assert report.rows_sampled == 30
    broken = "def repair(df):\n    raise ValueError('nope')\n"
    failed = patch.model_copy(update={"diff": unified_diff("remediations/x.py", broken)})
    failed_case = case.model_copy(update={"patch": failed})
    failed_update = run_pre_critic(
        MeridianState(run_id=uuid4(), partitions=[ref], cases={str(incident.id): failed_case}),
        num_cpus=1,
        timeout_s=30,
        memory_mb=512,
    )
    failed_report = failed_update["cases"][str(incident.id)].dry_run
    assert failed_report is not None
    assert failed_report.rows_sampled == 0
    missing = ref.model_copy(update={"parquet_path": str(tmp_path / "missing.parquet")})
    loaded = run_pre_critic(
        MeridianState(
            run_id=uuid4(),
            partitions=[missing],
            cases={str(incident.id): case},
        ),
        num_cpus=1,
        timeout_s=5,
        memory_mb=64,
    )
    detail = loaded["cases"][str(incident.id)].dry_run.detail
    assert "missing.parquet" in detail


def test_quarantine_and_row_edges(tmp_path: Path) -> None:
    before = pd.DataFrame({"speed_mps": [1.0, 2.0, 3.0]})
    after = pd.DataFrame({"speed_mps": [1.0]})
    assert 1 in changed_positions(before, after, quarantine=False)
    assert 2 in changed_positions(before, after, quarantine=True)
    nan_before = pd.DataFrame({"speed_mps": [float("nan"), None]})
    nan_after = pd.DataFrame({"speed_mps": [1.0, 1.0]})
    assert changed_positions(nan_before, nan_after, quarantine=False) == {0, 1}
    blank = pd.DataFrame({"note": [pd.NA, "kept"]})
    assert 0 in changed_positions(blank, pd.DataFrame({"note": ["x", "kept"]}), quarantine=False)
    ref = write_partition(
        tmp_path,
        dataset="vehicle_telemetry",
        fault=None,
        intensity=0,
        seed=2,
        rows=8,
        name="clean",
    )
    profile = ReferenceProfile.model_validate_json(
        Path(ref.reference_profile_path).read_text(encoding="utf-8")
    )
    assert _psi(pd.DataFrame({"other": [1.0]}), profile) == "under threshold"
    assert _psi(pd.DataFrame({"speed_mps": ["a", "b"]}), profile) == "under threshold"
    assert _contracts("vehicle_telemetry", before, "not-metadata", profile) == 0
    meta = PartitionMetadata(
        dataset="vehicle_telemetry",
        partition_date=DEFAULT_PARTITION_DATE,
        arrived_at=late_arrival(DEFAULT_PARTITION_DATE, 0.5),
    )
    assert _contracts("missing-dataset", before, meta, profile) == 0
    empty = _measured(
        before,
        before,
        "vehicle_telemetry",
        "nope",
        meta,
        profile,
        quarantine=False,
        detail="",
    )
    assert empty.detail == "dry-run inputs were not typed"
    with pytest.raises(ValueError, match=r"metadata.dataset"):
        failure_positions("other", before, meta, profile, FaultType.UNIT_MISMATCH)
    foreign = profile.model_copy(update={"dataset": "orders"})
    with pytest.raises(ValueError, match=r"reference_profile.dataset"):
        failure_positions(
            "vehicle_telemetry",
            before,
            meta,
            foreign,
            FaultType.UNIT_MISMATCH,
        )
    assert (
        failure_positions("vehicle_telemetry", before, meta, profile, FaultType.DUPLICATE_KEYS)
        == set()
    )
    late = failure_positions(
        "vehicle_telemetry",
        before,
        meta,
        profile,
        FaultType.LATE_PARTITION,
    )
    assert late == set()
