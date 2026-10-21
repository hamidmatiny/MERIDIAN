"""Submit-tool retries, and JSON mode when that is the selected method."""

from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict
from tests.contracts.builders import make_hypothesis, make_incident, make_patch

from meridian.agents.critic import run_critic
from meridian.agents.fix import PatchDraft, run_fix
from meridian.agents.root_cause import LoopState, RootCauseDraft, _force_answer, run_root_cause
from meridian.agents.triage import run_triage
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.severity import Severity
from meridian.contracts.state import MeridianState
from meridian.contracts.triage import TriageResult
from meridian.llm.errors import RunAborted
from meridian.llm.submit import output_method_of, run_submit_only, run_tool_loop


class Tiny(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["SEV1", "SEV2", "SEV3", "SEV4"]


class _Script:
    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.bound: list[list[str]] = []
        self.seen: list[list[object]] = []

    def bind_tools(self, tools: list[Any], **kwargs: object) -> "_Script":
        del kwargs
        self.bound.append([tool.name for tool in tools])
        return self

    def invoke(self, messages: list[object], **kwargs: object) -> object:
        del kwargs
        self.seen.append(list(messages))
        return self.replies.pop(0)


def _call(name: str, args: dict[str, object], call_id: str = "c") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _texts(script: _Script) -> str:
    chunks: list[str] = []
    for batch in script.seen:
        for message in batch:
            content = getattr(message, "content", "")
            chunks.append(content if isinstance(content, str) else str(content))
    return "\n".join(chunks)


def test_a_valid_submit_is_accepted() -> None:
    script = _Script([_call("submit_note", {"severity": "SEV2"})])
    parsed, calls = run_submit_only(
        script,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        node="triage",
        model_name="qwen",
        prompt_version="v1",
    )
    assert isinstance(parsed, Tiny)
    assert parsed.severity == "SEV2"
    assert calls[0].succeeded is True
    assert calls[0].structured is True
    assert script.bound == [["submit_note"]]
    assert output_method_of(object()) == "json_schema"
    marked = type("Marked", (), {"output_method": "submit_tool"})()
    assert output_method_of(marked) == "submit_tool"
    assert output_method_of(type("Other", (), {"output_method": "xml"})()) == "json_schema"


def test_invalid_args_are_returned_and_then_retried() -> None:
    script = _Script(
        [
            _call("submit_note", {"severity": 1}, "bad"),
            _call("submit_note", {"severity": "SEV1"}, "good"),
        ]
    )
    parsed, calls = run_submit_only(
        script,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        node="triage",
        model_name="qwen",
        prompt_version="v1",
    )
    assert isinstance(parsed, Tiny)
    assert parsed.severity == "SEV1"
    assert calls[0].succeeded is False
    assert "severity" in _texts(script)
    assert len(calls) == 2


def test_plain_text_gets_one_corrective_retry() -> None:
    script = _Script(
        [
            AIMessage(content="I think it is SEV2"),
            _call("submit_note", {"severity": "SEV2"}),
        ]
    )
    parsed, calls = run_submit_only(
        script,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        node="triage",
        model_name="qwen",
        prompt_version="v1",
    )
    assert isinstance(parsed, Tiny)
    assert "Call the submit tool. Do not reply with plain text." in _texts(script)
    assert len(calls) == 2


def test_a_second_plain_text_reply_degrades() -> None:
    script = _Script([AIMessage(content="prose"), AIMessage(content="still prose")])
    parsed, calls = run_submit_only(
        script,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        node="triage",
        model_name="qwen",
        prompt_version="v1",
    )
    assert parsed is None
    assert len(calls) == 2
    assert all(call.succeeded is False for call in calls)


def test_two_validation_retries_then_degrade() -> None:
    script = _Script([_call("submit_note", {}) for _index in range(3)])
    parsed, calls = run_submit_only(
        script,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        node="triage",
        model_name="qwen",
        prompt_version="v1",
    )
    assert parsed is None
    assert len(calls) == 3


def test_budget_exhaustion_forces_a_submit() -> None:
    def read_log(path: str) -> str:
        return f'{{"ref": "{path}"}}'

    tool = StructuredTool.from_function(read_log, name="read_log", description="Read a log.")
    script = _Script(
        [
            _call("read_log", {"path": "events.jsonl"}, "tool-1"),
            _call("submit_note", {"severity": "SEV3"}, "submit-1"),
        ]
    )
    parsed, calls, records, payloads = run_tool_loop(
        script,
        [HumanMessage(content="investigate")],
        Tiny,
        tool_name="submit_note",
        mcp_tools=[tool],
        node="root_cause",
        model_name="qwen",
        prompt_version="v1",
        max_tool_calls=1,
    )
    assert isinstance(parsed, Tiny)
    assert script.bound[0] == ["read_log", "submit_note"]
    assert script.bound[1] == ["submit_note"]
    assert "submit now" in _texts(script)
    assert records[0].ok is True
    assert payloads[0] == '{"ref": "events.jsonl"}'
    assert calls[-1].succeeded is True
    assert calls[0].structured is False


def test_an_unknown_tool_is_recorded_and_a_raising_tool_is_not_ok() -> None:
    def boom() -> str:
        raise RuntimeError("nope")

    tool = StructuredTool.from_function(boom, name="boom", description="Fail.")
    script = _Script(
        [
            _call("missing", {}, "m"),
            _call("boom", {}, "b"),
            _call("submit_note", {"severity": "SEV4"}, "s"),
        ]
    )
    parsed, _calls, records, payloads = run_tool_loop(
        script,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        mcp_tools=[tool],
        node="fix",
        model_name="qwen",
        prompt_version="v1",
        max_tool_calls=2,
    )
    assert isinstance(parsed, Tiny)
    assert records[0].ok is False
    assert payloads[0].startswith("unknown tool")
    assert records[1].ok is False
    assert "nope" in payloads[1]


def test_json_mode_retries_a_parse_error() -> None:
    class _Json:
        output_method = "json_mode"

        def __init__(self) -> None:
            self.methods: list[object] = []
            self.seen: list[object] = []

        def with_structured_output(self, schema: type[BaseModel], **kwargs: object) -> "_Json":
            self.methods.append(kwargs.get("method"))
            self.schema = schema
            return self

        def invoke(self, messages: list[object], **kwargs: object) -> dict[str, object]:
            del kwargs
            self.seen.append(messages)
            if len(self.seen) == 1:
                return {"raw": None, "parsed": None, "parsing_error": ValueError("bad json")}
            return {
                "raw": None,
                "parsed": self.schema(severity="SEV2"),
                "parsing_error": None,
            }

    chat = _Json()
    parsed, calls = run_submit_only(
        chat,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        node="critic",
        model_name="gemma",
        prompt_version="v1",
    )
    assert isinstance(parsed, Tiny)
    assert chat.methods == ["json_mode", "json_mode"]
    assert calls[-1].succeeded is True
    assert "bad json" in str(chat.seen[1][-1].content)


def test_json_mode_degrades_after_two_retries() -> None:
    class _Json:
        output_method = "json_mode"

        def with_structured_output(self, schema: object, **kwargs: object) -> "_Json":
            del schema, kwargs
            return self

        def invoke(self, messages: object, **kwargs: object) -> dict[str, object]:
            del messages, kwargs
            return {"raw": None, "parsed": {"severity": 1}, "parsing_error": None}

    parsed, calls = run_submit_only(
        _Json(),
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        node="critic",
        model_name="gemma",
        prompt_version="v1",
    )
    assert parsed is None
    assert len(calls) == 3


def test_fix_json_mode_keeps_a_valid_draft() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    case = IncidentCase(incident=incident, hypothesis=hypothesis)

    class _Json:
        output_method = "json_mode"

        def bind_tools(self, tools: object, **kwargs: object) -> "_Json":
            del tools, kwargs
            return self

        def with_structured_output(self, schema: object, **kwargs: object) -> "_Json":
            assert kwargs.get("method") == "json_mode"
            del schema
            return self

        def invoke(self, messages: object, **kwargs: object) -> object:
            del kwargs
            if not isinstance(messages, list) or len(messages) < 2:
                return AIMessage(content="")
            return {
                "raw": None,
                "parsed": PatchDraft(
                    remediation_kind="QUARANTINE",
                    module_source="def quarantine(df):\n    return df.iloc[0:0]\n",
                    regression_test="def check(df):\n    return None\n",
                ),
                "parsing_error": None,
            }

    state = run_fix(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        model_name="gemma",
        chat=_Json(),
        tools=[],
        max_tool_calls=1,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    patch = state["cases"][str(incident.id)].patch
    assert patch is not None
    assert patch.remediation_kind is RemediationKind.QUARANTINE


def test_root_json_mode_marks_a_failed_answer_as_the_budget_fallback() -> None:
    class _Json:
        output_method = "json_mode"

        def with_structured_output(self, schema: object, **kwargs: object) -> "_Json":
            assert kwargs.get("method") == "json_mode"
            del schema
            return self

        def invoke(self, messages: object, **kwargs: object) -> dict[str, object]:
            del messages, kwargs
            return {"raw": None, "parsed": None, "parsing_error": ValueError("no")}

    failed = _force_answer(
        LoopState(incident_id=uuid4(), prompt_version="v1", turns=[]),
        _Json(),  # type: ignore[arg-type]
        "gemma",
        "v1",
    )
    assert failed["budget_fallback"] is True

    class _Ok(_Json):
        def invoke(self, messages: object, **kwargs: object) -> dict[str, object]:
            del messages, kwargs
            return {
                "raw": None,
                "parsed": RootCauseDraft(root_cause_summary="the scale changed", confidence=0.4),
                "parsing_error": None,
            }

    held = _force_answer(
        LoopState(incident_id=uuid4(), prompt_version="v1", turns=[]),
        _Ok(),  # type: ignore[arg-type]
        "gemma",
        "v1",
    )
    assert held["budget_fallback"] is False


class _BoundChat:
    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.output_method = "submit_tool"

    def bind_tools(self, tools: list[Any], **kwargs: object) -> "_BoundChat":
        del tools, kwargs
        return self

    def invoke(self, messages: object, **kwargs: object) -> object:
        del messages, kwargs
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _case(incident: Any) -> MeridianState:
    return MeridianState(
        run_id=uuid4(),
        cases={str(incident.id): IncidentCase(incident=incident)},
    )


def test_triage_submit_accepts_and_then_degrades(tmp_path: Path) -> None:
    del tmp_path
    incident = make_incident()
    good = {
        "incident_id": str(incident.id),
        "severity": "SEV2",
        "rationale": "the null rate jumped",
        "affected_consumers": ["trip_summary"],
        "confidence": 0.4,
    }
    accepted = run_triage(
        _case(incident),
        model_name="qwen",
        chat=_BoundChat([_call("submit_triage_assessment", good)]),
    )
    triage = accepted["cases"][str(incident.id)].triage
    assert triage is not None
    assert triage.llm_severity is Severity.SEV2
    assert triage.triage_degraded is False
    wrong = dict(good)
    wrong["incident_id"] = str(uuid4())
    degraded = run_triage(
        _case(incident),
        model_name="qwen",
        chat=_BoundChat([_call("submit_triage_assessment", wrong)]),
    )
    assert degraded["cases"][str(incident.id)].triage.triage_degraded is True
    stopped = MeridianState(
        run_id=uuid4(),
        aborted=True,
        abort_reason="BILLING",
        cases=_case(incident).cases,
    )
    assert run_triage(stopped, model_name="qwen", chat=_BoundChat([]))["aborted"] is True
    aborted = run_triage(
        _case(incident),
        model_name="qwen",
        chat=_BoundChat([RunAborted("UNSUPPORTED", "down")]),
    )
    assert aborted["abort_reason"].startswith("UNSUPPORTED")


def test_critic_submit_records_a_verdict() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    case = IncidentCase(incident=incident, hypothesis=hypothesis, patch=patch)
    state = MeridianState(run_id=uuid4(), cases={str(incident.id): case})
    updated = run_critic(
        state,
        model_name="gemma",
        chat=_BoundChat([_call("submit_verdict", {"approved": False, "objections": []})]),
    )
    verdicts = updated["cases"][str(incident.id)].verdicts
    assert verdicts[0].approved is False
    assert verdicts[0].critic_degraded is False
    assert (
        run_critic(
            state.model_copy(update={"aborted": True, "abort_reason": "AUTH"}),
            model_name="gemma",
            chat=_BoundChat([]),
        )["aborted"]
        is True
    )
    failed = run_critic(
        state,
        model_name="gemma",
        chat=_BoundChat([RunAborted("UNSUPPORTED", "down")]),
    )
    assert "UNSUPPORTED" in str(failed["abort_reason"])


def test_root_and_fix_submit_on_the_first_turn(tmp_path: Path) -> None:
    incident = make_incident()
    triage = TriageResult(
        incident_id=incident.id,
        severity=Severity.SEV2,
        rationale="floor kept",
        affected_consumers=["trip_summary"],
        confidence=0.2,
        llm_severity=Severity.SEV2,
        floor_severity=Severity.SEV2,
        final_severity=Severity.SEV2,
        prompt_version="v1",
    )
    case = IncidentCase(incident=incident, triage=triage)
    found = run_root_cause(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        model_name="qwen",
        chat=_BoundChat(
            [
                _call(
                    "submit_hypothesis",
                    {"root_cause_summary": "a scale factor changed", "confidence": 0.6},
                )
            ]
        ),
        tools=[],
        max_tool_calls=2,
        evidence_root=tmp_path,
    )
    saved = found["cases"][str(incident.id)]
    assert saved.hypothesis is not None
    assert saved.hypothesis_from_budget_fallback is False
    missed = run_root_cause(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        model_name="qwen",
        chat=_BoundChat([AIMessage(content="prose"), AIMessage(content="still prose")]),
        tools=[],
        max_tool_calls=2,
        evidence_root=tmp_path,
    )
    assert missed["cases"][str(incident.id)].hypothesis_from_budget_fallback is True
    stopped = MeridianState(
        run_id=uuid4(),
        aborted=True,
        abort_reason="QUOTA",
        cases={str(incident.id): case},
    )
    assert (
        run_root_cause(
            stopped,
            model_name="qwen",
            chat=_BoundChat([]),
            tools=[],
            max_tool_calls=1,
            evidence_root=tmp_path,
        )["aborted"]
        is True
    )
    hypothesis = make_hypothesis(incident_id=incident.id)
    fix_case = IncidentCase(incident=incident, hypothesis=hypothesis)
    fixed = run_fix(
        MeridianState(run_id=uuid4(), cases={str(incident.id): fix_case}),
        model_name="qwen",
        chat=_BoundChat(
            [
                _call("sample_failures", {"incident_id": str(incident.id), "n": 1}, "sample"),
                _call(
                    "submit_patch",
                    {
                        "remediation_kind": "DATA_REPAIR",
                        "module_source": "def repair(df):\n    return df\n",
                        "regression_test": "def check(df):\n    return None\n",
                    },
                    "patch",
                ),
            ]
        ),
        tools=[],
        max_tool_calls=2,
        max_row_loss=0.01,
        max_quarantine=0.25,
    )
    assert fixed["cases"][str(incident.id)].patch.remediation_kind is RemediationKind.DATA_REPAIR
    halted = MeridianState(
        run_id=uuid4(),
        aborted=True,
        abort_reason="BILLING",
        cases={str(incident.id): fix_case},
    )
    assert (
        run_fix(
            halted,
            model_name="qwen",
            chat=_BoundChat([]),
            tools=[],
            max_tool_calls=1,
            max_row_loss=0.01,
            max_quarantine=0.25,
        )["aborted"]
        is True
    )


def test_the_last_turn_rejects_a_read_tool_and_then_submits() -> None:
    script = _Script(
        [
            _call("read_log", {"path": "a"}, "mcp"),
            _call("submit_note", {}, "bad"),
            _call("submit_note", {"severity": "SEV1"}, "good"),
        ]
    )
    parsed, _calls, _records, _payloads = run_tool_loop(
        script,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        mcp_tools=[],
        node="fix",
        model_name="qwen",
        prompt_version="v1",
        max_tool_calls=0,
    )
    assert isinstance(parsed, Tiny)
    assert script.bound[0] == ["submit_note"]
    assert "submit now" in _texts(script)


def test_a_non_dict_tool_call_is_treated_as_text() -> None:
    script = _Script(
        [
            type("Odd", (), {"tool_calls": ["nope"], "content": ""})(),
            type(
                "Blank",
                (),
                {
                    "tool_calls": [
                        {"name": "", "args": {"severity": "SEV2"}, "id": "blank"},
                    ],
                    "content": "",
                },
            )(),
        ]
    )
    parsed, _calls = run_submit_only(
        script,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        node="triage",
        model_name="qwen",
        prompt_version="v1",
    )
    assert parsed is None


def test_the_last_turn_degrades_when_text_persists() -> None:
    script = _Script([AIMessage(content="no"), AIMessage(content="still no")])
    parsed, calls, _records, _payloads = run_tool_loop(
        script,
        [HumanMessage(content="go")],
        Tiny,
        tool_name="submit_note",
        mcp_tools=[],
        node="fix",
        model_name="qwen",
        prompt_version="v1",
        max_tool_calls=0,
    )
    assert parsed is None
    assert len(calls) == 2


def test_the_critic_corrective_names_submit_verdict() -> None:
    from meridian.llm.submit import CRITIC_CORRECTIVE

    class Decision(BaseModel):
        model_config = ConfigDict(extra="ignore")

        approved: bool = False

    script = _Script(
        [
            AIMessage(content="looks fine"),
            _call("submit_verdict", {"approved": True, "objections": []}),
        ]
    )
    parsed, calls = run_submit_only(
        script,
        [HumanMessage(content="go")],
        Decision,
        tool_name="submit_verdict",
        node="critic",
        model_name="gemma",
        prompt_version="v1",
        text_retry=CRITIC_CORRECTIVE,
    )
    assert parsed is not None
    assert CRITIC_CORRECTIVE in _texts(script)
    assert calls[1].corrective is True
    assert calls[0].corrective is False


def test_a_data_repair_without_sample_failures_is_a_tool_error() -> None:
    from meridian.agents.fix import PatchDraft, _require_sample

    repair = {
        "remediation_kind": "DATA_REPAIR",
        "module_source": "def repair(df):\n    return df\n",
        "regression_test": "def check(df):\n    return None\n",
    }
    script = _Script(
        [
            _call("submit_patch", repair, "one"),
            _call("sample_failures", {"incident_id": "x", "n": 1}, "two"),
            _call("submit_patch", repair, "three"),
        ]
    )
    parsed, _calls, records, _payloads = run_tool_loop(
        script,
        [HumanMessage(content="go")],
        PatchDraft,
        tool_name="submit_patch",
        mcp_tools=[],
        node="fix",
        model_name="qwen",
        prompt_version="v1",
        max_tool_calls=4,
        submit_guard=_require_sample,
    )
    assert isinstance(parsed, PatchDraft)
    assert any(record.tool == "sample_failures" for record in records)
    assert "sample_failures" in _texts(script)


def test_retry_feedback_includes_the_previous_dry_run() -> None:
    from meridian.agents.fix import _feedback
    from meridian.contracts.dry_run import DryRunReport
    from meridian.contracts.incident_case import IncidentCase

    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    case = IncidentCase(
        incident=incident,
        hypothesis=hypothesis,
        patch=make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id),
        dry_run=DryRunReport(
            rows_sampled=10,
            rows_changed=4,
            rows_changed_outside=3,
            over_broad=True,
            contracts_after=1,
            rows_fixed=2,
            psi_after="0.4",
        ),
    )
    text = _feedback(case)
    assert "rows_changed_outside=3" in text
    assert "rows_fixed=2" in text
    assert "over_broad=True" in text
    assert "psi_after=0.4" in text
