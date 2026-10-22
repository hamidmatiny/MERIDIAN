"""Branches the loop nodes take when the model or the partition is unusable."""

from uuid import uuid4

import pytest
from tests.contracts.builders import make_hypothesis, make_incident, make_patch

from meridian.agents.critic import VerdictDraft, run_critic
from meridian.agents.fix import run_fix
from meridian.agents.verifier import run_verifier
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.state import MeridianState
from meridian.graph.limits import (
    record_verdict,
    route_after_critic,
    route_after_root,
    route_after_verifier,
)


def _state(*cases: IncidentCase) -> MeridianState:
    return MeridianState(run_id=uuid4(), cases={str(case.incident.id): case for case in cases})


def test_fix_and_critic_reject_the_wrong_branch_shape() -> None:
    incident = make_incident()
    bare = IncidentCase(incident=incident)
    with pytest.raises(ValueError, match="one incident"):
        run_fix(
            _state(bare, IncidentCase(incident=make_incident())),
            model_name="x",
            chat=object(),
            tools=[],
            max_tool_calls=1,
            max_row_loss=0.01,
            max_quarantine=0.25,
        )
    with pytest.raises(ValueError, match="hypothesis"):
        run_fix(
            _state(bare),
            model_name="x",
            chat=object(),
            tools=[],
            max_tool_calls=1,
            max_row_loss=0.01,
            max_quarantine=0.25,
        )
    hypothesis = make_hypothesis(incident_id=incident.id)
    patched = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id),
        needs_human=True,
    )
    with pytest.raises(ValueError, match="one incident"):
        run_critic(
            _state(patched, IncidentCase(incident=make_incident())),
            model_name="x",
            chat=object(),
        )
    with pytest.raises(ValueError, match="patch"):
        run_critic(
            _state(IncidentCase(incident=incident, hypothesis=hypothesis)),
            model_name="x",
            chat=object(),
        )


def test_fix_records_an_unknown_tool_and_a_tool_error() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    case = IncidentCase(incident=incident, hypothesis=hypothesis)

    class _Tool:
        name = "list_events"

        def invoke(self, _args: dict[str, object]) -> str:
            raise RuntimeError("tool down")

    class _Chat:
        def bind_tools(self, tools: object, **_kwargs: object) -> "_Bound":
            del tools
            return _Bound()

        def with_structured_output(self, schema: object, **_kwargs: object) -> "_Bound":
            del schema
            return _Bound(structured=True)

    class _Bound:
        def __init__(self, *, structured: bool = False) -> None:
            self.structured = structured
            self.calls = 0

        def invoke(self, messages: object, **_kwargs: object) -> object:
            del messages
            self.calls += 1
            if self.structured:
                draft = {
                    "remediation_kind": "NOT_A_KIND",
                    "module_source": " ",
                    "regression_test": " ",
                }
                return {"raw": None, "parsed": draft, "parsing_error": None}
            if self.calls == 1:
                return type(
                    "Msg",
                    (),
                    {
                        "tool_calls": [
                            {"name": "missing", "args": {"n": 1}, "id": ""},
                            "not-a-dict",
                        ]
                    },
                )()
            if self.calls == 2:
                return type(
                    "Msg",
                    (),
                    {"tool_calls": [{"name": "list_events", "args": {}, "id": "c"}]},
                )()
            raise RuntimeError("planner down")

    state = run_fix(
        _state(case),
        model_name="script",
        chat=_Chat(),
        tools=[_Tool()],  # type: ignore[list-item]
        max_tool_calls=4,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    updated = state["cases"][str(incident.id)]
    assert updated.patch is not None
    assert updated.patch.remediation_kind is RemediationKind.ESCALATE
    assert state["tool_calls"][0].ok is False
    assert any(not call.succeeded for call in state["llm_calls"])


def test_critic_accepts_a_dict_draft_and_drops_a_blank_objection() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    case = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        needs_human=True,
    )

    class _Chat:
        def with_structured_output(self, schema: object, **_kwargs: object) -> "_Chat":
            del schema
            return self

        def invoke(self, messages: object, **_kwargs: object) -> dict[str, object]:
            del messages
            return {
                "raw": type(
                    "Raw",
                    (),
                    {"usage_metadata": {"input_tokens": True, "output_tokens": -3}},
                )(),
                "parsed": {
                    "approved": False,
                    "objections": [
                        {"kind": "NOT_A_KIND", "message": "unknown"},
                        {"kind": "OTHER", "message": "  "},
                    ],
                },
                "parsing_error": None,
            }

    state = run_critic(_state(case), model_name="grok", chat=_Chat())
    verdict = state["cases"][str(incident.id)].verdicts[-1]
    assert verdict.approved is False
    assert verdict.objections[0].kind.value == "OTHER"
    assert state["llm_calls"][0].input_tokens == 0


def test_critic_rejects_a_non_uuid_patch_id() -> None:
    from meridian.agents.critic import _verdict

    with pytest.raises(TypeError):
        _verdict(VerdictDraft(approved=True), "nope", "grok", 1)


def test_route_helpers_stop_finished_cases() -> None:
    incident = make_incident()
    case = IncidentCase(incident=incident)
    assert route_after_root(case) == "end"
    assert route_after_critic(case) == "end"
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    reviewed = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        fix_attempts=1,
        needs_human=True,
    )
    from meridian.contracts.verdict import Verdict

    with pytest.raises(ValueError, match="next critic round"):
        record_verdict(
            reviewed,
            Verdict(patch_id=patch.id, reviewer_model="grok", approved=True, round=2),
        )
    capped = reviewed.model_copy(update={"fix_attempts": 3, "verification": None})
    assert route_after_verifier(capped) == "end"


def test_verifier_rejects_a_bad_branch_and_a_missing_file(tmp_path) -> None:
    incident = make_incident()
    other = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    case = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=patch,
        needs_human=True,
    )
    with pytest.raises(ValueError, match="one incident"):
        run_verifier(
            _state(case, IncidentCase(incident=other)),
            num_cpus=1,
            timeout_s=1,
            memory_mb=1,
            max_row_loss=0.01,
            max_quarantine=0.25,
        )
    with pytest.raises(ValueError, match="patch"):
        run_verifier(
            _state(IncidentCase(incident=incident, hypothesis=hypothesis)),
            num_cpus=1,
            timeout_s=1,
            memory_mb=1,
            max_row_loss=0.01,
            max_quarantine=0.25,
        )
    from tests.agents.support import write_partition

    from meridian.agents.patch_text import unified_diff
    from meridian.contracts.fault_type import FaultType
    from meridian.contracts.patch import Patch

    ref = write_partition(
        tmp_path,
        dataset="fleet_daily_kpi",
        fault=FaultType.LATE_PARTITION,
        intensity=0.1,
        seed=2,
        rows=20,
        name="late",
    )
    missing = ref.model_copy(update={"parquet_path": str(tmp_path / "absent.parquet")})
    bad = Patch(
        id=patch.id,
        incident_id=incident.id,
        hypothesis_id=hypothesis.id,
        remediation_kind=RemediationKind.DATA_REPAIR,
        diff=unified_diff("remediations/x.py", "import os\ndef repair(df):\n    return df\n"),
        regression_test="def check(df):\n    return None\n",
        files_touched=["remediations/x.py"],
        author_model="test",
    )
    # The incident dataset will not match fleet, so use the partition's dataset via a real incident.
    from pathlib import Path

    from meridian.contracts.reference_profile import ReferenceProfile
    from meridian.pipeline.parquet import read_parquet
    from meridian.quality.evaluate import evaluate

    profile = ReferenceProfile.model_validate_json(Path(ref.reference_profile_path).read_text())
    found = evaluate(ref.dataset, read_parquet(Path(ref.parquet_path)), ref.metadata, profile)
    incident = next(item for item in found if item.fault_type is FaultType.LATE_PARTITION)
    hypothesis = make_hypothesis(incident_id=incident.id)
    bad = bad.model_copy(update={"incident_id": incident.id, "hypothesis_id": hypothesis.id})
    loaded = IncidentCase(incident=incident, hypothesis=hypothesis, patch=bad)
    updated = run_verifier(
        MeridianState(run_id=uuid4(), partitions=[missing], cases={str(incident.id): loaded}),
        num_cpus=1,
        timeout_s=5,
        memory_mb=128,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    result = next(iter(updated["cases"].values())).verification
    assert result is not None
    assert result.passed is False
