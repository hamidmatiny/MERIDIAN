"""The bench scorer may see the clean frame. It does not claim an accuracy."""

from uuid import uuid4

import numpy as np
import pandas as pd
from tests.contracts.builders import make_hypothesis

from meridian.agents.fake_fix import _REPAIR
from meridian.agents.patch_text import unified_diff
from meridian.bench.score_fix import score
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.patch import Patch
from meridian.contracts.remediation import RemediationKind
from meridian.injector.faults import INJECTORS
from meridian.pipeline.generate import generate_dataset


def _case(clean, fault, kind: RemediationKind, source: str) -> tuple[IncidentCase, object]:
    _frame, label = INJECTORS[fault](
        clean,
        np.random.default_rng(4),
        0.1,
        dataset="vehicle_telemetry",
        seed=4,
    )
    from datetime import UTC, datetime

    from meridian.contracts.incident import Incident

    incident = Incident(
        id=uuid4(),
        detected_at=datetime(2026, 10, 20, tzinfo=UTC),
        dataset="vehicle_telemetry",
        partition="2026-10-19",
        contract_name="VehicleTelemetry",
        fault_type=fault,
        failing_checks=["check"],
    )
    hypothesis = make_hypothesis(incident_id=incident.id)
    path = f"remediations/{incident.id}.py"
    patch = Patch(
        id=uuid4(),
        incident_id=incident.id,
        hypothesis_id=hypothesis.id,
        remediation_kind=kind,
        diff=unified_diff(path, source),
        regression_test="def check(df):\n    return None\n",
        files_touched=[path],
        author_model="test",
    )
    return IncidentCase(incident=incident, hypothesis=hypothesis, patch=patch), label


def test_escalate_and_rerun_do_not_claim_a_row_match() -> None:
    from meridian.contracts.fault_type import FaultType

    clean = generate_dataset("vehicle_telemetry", 4, rows=20)
    escalated, label = _case(
        clean,
        FaultType.NULL_EXPLOSION,
        RemediationKind.ESCALATE,
        "def repair(df):\n    return df\n",
    )
    # Rebuild the label's fault: the helper injected NULL even for this call.
    scored = score(escalated, clean, label)
    assert scored.kind_appropriate is True
    assert scored.repaired_matches_clean is False
    assert scored.rows_lost_pct == 0.0

    rerun, late = _case(
        clean,
        FaultType.LATE_PARTITION,
        RemediationKind.RERUN_UPSTREAM,
        "def repair(df):\n    return df\n",
    )
    scored = score(rerun, clean, late)
    assert scored.kind_appropriate is True
    null_rerun, null_label = _case(
        clean,
        FaultType.NULL_EXPLOSION,
        RemediationKind.RERUN_UPSTREAM,
        "def repair(df):\n    return df\n",
    )
    assert score(null_rerun, clean, null_label).kind_appropriate is True
    assert scored.repaired_matches_clean is True
    assert scored.rows_lost_pct == 0.0


def test_unit_repair_matches_the_clean_frame() -> None:
    from meridian.contracts.fault_type import FaultType

    clean = generate_dataset("vehicle_telemetry", 4, rows=30)
    case, label = _case(clean, FaultType.UNIT_MISMATCH, RemediationKind.DATA_REPAIR, _REPAIR)
    scored = score(case, clean, label)
    assert scored.kind_appropriate is True
    assert scored.repaired_matches_clean is True
    assert scored.rows_lost_pct == 0.0


def test_quarantine_score_uses_percentage_points() -> None:
    from meridian.contracts.fault_type import FaultType

    clean = generate_dataset("vehicle_telemetry", 4, rows=40)
    case, label = _case(
        clean,
        FaultType.NULL_EXPLOSION,
        RemediationKind.QUARANTINE,
        "def quarantine(df):\n    return df.isna().any(axis=1)\n",
    )
    scored = score(case, clean, label)
    assert scored.kind_appropriate is True
    assert scored.rows_lost_pct > 1.0


def test_rejected_source_does_not_match() -> None:
    from meridian.bench.score_fix import _frames_match
    from meridian.contracts.fault_type import FaultType

    clean = generate_dataset("vehicle_telemetry", 4, rows=20)
    case, label = _case(
        clean,
        FaultType.UNIT_MISMATCH,
        RemediationKind.DATA_REPAIR,
        "import os\ndef repair(df):\n    return df\n",
    )
    scored = score(case, clean, label)
    assert scored.repaired_matches_clean is False
    left = pd.DataFrame({"name": ["a"], "speed_mps": [1.0]})
    right = pd.DataFrame({"name": ["b"], "speed_mps": [2.0]})
    assert _frames_match(left, right) is False
    assert _frames_match(left, pd.DataFrame({"other": [1]})) is False


def test_missing_patch_is_not_appropriate() -> None:
    from datetime import UTC, datetime

    from meridian.contracts.fault_type import FaultType
    from meridian.contracts.incident import Incident

    clean = generate_dataset("vehicle_telemetry", 4, rows=10)
    _frame, label = INJECTORS[FaultType.UNIT_MISMATCH](
        clean,
        np.random.default_rng(4),
        0.1,
        dataset="vehicle_telemetry",
        seed=4,
    )
    incident = Incident(
        id=uuid4(),
        detected_at=datetime(2026, 10, 20, tzinfo=UTC),
        dataset="vehicle_telemetry",
        partition="2026-10-19",
        contract_name="VehicleTelemetry",
        fault_type=FaultType.UNIT_MISMATCH,
        failing_checks=["check"],
    )
    case = IncidentCase(incident=incident, hypothesis=make_hypothesis(incident_id=incident.id))
    scored = score(case, clean, label)
    assert scored.kind_appropriate is False
    assert scored.repaired_matches_clean is False
