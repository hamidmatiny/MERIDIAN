"""Recorded timelines, the Slack audit log, and diff uploads."""

import hashlib
import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest
from tests.contracts.builders import DIFF, make_approval, make_incident, make_patch

from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.progress_event import ProgressEvent
from meridian.contracts.remediation import RemediationKind
from meridian.progress import load_progress, node_line
from meridian.slack.audit import AuditLog
from meridian.slack.blocks import decided_blocks
from meridian.slack.gate import ApprovalPorts, _post, _show, _upload_diff
from meridian.slack.gateway import RecordingGateway
from meridian.slack.handlers import (
    ALREADY_DECIDED,
    NOT_AUTHORIZED,
    on_action,
    on_mention,
    on_reject_submit,
)
from meridian.slack.report import TIMES_NOT_RECORDED, build_postmortem, build_timeline
from meridian.slack.store import DecisionLog, MessageLog, OpenCase, OpenIndex
from meridian.slack.text import mention_user

_NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)


def _event(
    incident_id: str,
    run_id: UUID,
    node: str,
    phase: Literal["start", "end"],
    at: datetime,
    attempt: int = 1,
) -> ProgressEvent:
    return ProgressEvent(
        occurred_at=at,
        run_id=run_id,
        case_id=incident_id,
        node=node,
        attempt=attempt,
        phase=phase,
        elapsed_s=0.0,
    )


def test_timeline_uses_recorded_times_and_the_gaps_are_not_uniform() -> None:
    incident = make_incident(detected_at=_NOW)
    run_id = uuid4()
    incident_id = str(incident.id)
    moments = [
        _NOW + timedelta(seconds=4),
        _NOW + timedelta(seconds=19),
        _NOW + timedelta(seconds=47),
        _NOW + timedelta(seconds=100),
    ]
    events = [
        _event(incident_id, run_id, "sentinel", "end", moments[0]),
        _event(incident_id, run_id, "triage", "end", moments[1]),
        _event(incident_id, run_id, "fix", "start", moments[2], attempt=2),
        _event(incident_id, run_id, "slack_approval", "start", moments[3]),
    ]
    posted = moments[3] + timedelta(seconds=2)
    decided = posted + timedelta(seconds=80)
    approval = make_approval().model_copy(update={"decided_at": decided, "approver": "UOWNER"})
    case = IncidentCase(incident=incident, approval=approval)
    timeline = build_timeline(case, events)
    assert [item.occurred_at for item in timeline[:-1]] == moments
    assert timeline[-1].occurred_at == decided
    assert "<@UOWNER>" in timeline[-1].summary
    gaps = [
        (later.occurred_at - earlier.occurred_at).total_seconds()
        for earlier, later in itertools.pairwise(timeline)
    ]
    assert len(set(gaps)) > 1
    assert gaps != [1.0] * len(gaps)
    report = build_postmortem(case, [], events=events, posted_at=posted)
    assert report.machine_time_s == 100
    assert report.human_wait_s == 80
    assert report.time_to_resolution_s == 180


def test_a_run_without_node_times_is_marked_instead_of_invented() -> None:
    incident = make_incident(detected_at=_NOW)
    approval = make_approval().model_copy(update={"decided_at": _NOW + timedelta(hours=1)})
    case = IncidentCase(incident=incident, approval=approval)
    report = build_postmortem(case, [])
    assert report.machine_time_s is None
    assert report.human_wait_s is None
    assert report.time_to_resolution_s == 0
    assert report.timeline[0].summary == TIMES_NOT_RECORDED
    assert report.timeline[0].occurred_at == incident.detected_at
    assert "<@U123>" in report.timeline[1].summary


def test_node_line_round_trips_the_recorded_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MERIDIAN_PROGRESS_DIR", str(tmp_path))
    run_id = uuid4()
    node_line(run_id, "case-1", "triage", 1, 3.0, "start")
    node_line(run_id, "case-1", "triage", 1, 9.0, "end")
    node_line(run_id, "case-1", "triage", 1, 9.0, "nope")
    loaded = load_progress(run_id)
    assert [item.phase for item in loaded] == ["start", "end"]
    assert loaded[0].occurred_at <= loaded[1].occurred_at
    assert load_progress(uuid4()) == []


def test_each_audit_event_is_one_json_line_and_tokens_are_dropped(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "slack.jsonl")
    audit.write("mention_received", text="<@UBOT> status", user="UOWNER")
    audit.write("reply_sent", text="No open cases.")
    audit.write("message_posted", ts="1.0", kind="approval")
    audit.write("file_uploaded", name="a.diff", sha256="abc", bytes=4)
    audit.write("action_received", action="meridian_approve", user="UOWNER", patch_id="p1")
    audit.write("decision_recorded", action="APPROVED", user="UOWNER", patch_id="p1")
    audit.write("duplicate_rejected", action="meridian_approve", user="UOWNER", patch_id="p1")
    audit.write(
        "unauthorized_rejected",
        action="meridian_approve",
        user="UOTHER",
        patch_id="p1",
    )
    audit.write("message_updated", ts="1.0", buttons_removed=True)
    audit.write("reply_sent", text="prefix xoxb-supersecretvalue suffix", token="xoxb-another")
    names = [line["event"] for line in audit.lines()]
    assert names == [
        "mention_received",
        "reply_sent",
        "message_posted",
        "file_uploaded",
        "action_received",
        "decision_recorded",
        "duplicate_rejected",
        "unauthorized_rejected",
        "message_updated",
        "reply_sent",
    ]
    raw = (tmp_path / "slack.jsonl").read_text(encoding="utf-8")
    assert "xoxb-" not in raw
    assert "[redacted]" in raw
    assert "token" not in raw
    uploaded = audit.lines()[3]
    assert uploaded["bytes"] == 4
    assert uploaded["sha256"] == "abc"


def test_handlers_record_mention_reply_decision_and_rejections(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "slack.jsonl")
    said: list[str] = []
    on_mention(
        {"text": "<@UBOT> status", "user": "UOWNER", "ts": "9.0"},
        lambda **kwargs: said.append(str(kwargs["text"])),
        bot_user_id="UBOT",
        cases=[],
        now=_NOW,
        audit=audit,
    )
    assert said == ["No open cases."]
    log = DecisionLog(tmp_path / "decisions")
    value = json.dumps({"patch_id": "p1", "thread_id": "t1"})
    resumed: list[dict[str, object]] = []
    assert (
        on_action(
            user_id="UOTHER",
            action_id="meridian_approve",
            value=value,
            approvers=frozenset({"UOWNER"}),
            log=log,
            resume=lambda _thread, payload: resumed.append(payload),
            now=_NOW,
            channel="C1",
            message_ts="1",
            audit=audit,
        )
        == NOT_AUTHORIZED
    )
    assert (
        on_action(
            user_id="UOWNER",
            action_id="meridian_approve",
            value=value,
            approvers=frozenset({"UOWNER"}),
            log=log,
            resume=lambda _thread, payload: resumed.append(payload),
            now=_NOW,
            channel="C1",
            message_ts="1",
            audit=audit,
        )
        == ""
    )
    assert (
        on_action(
            user_id="UOWNER",
            action_id="meridian_approve",
            value=value,
            approvers=frozenset({"UOWNER"}),
            log=log,
            resume=lambda _thread, payload: resumed.append(payload),
            now=_NOW,
            channel="C1",
            message_ts="1",
            audit=audit,
        )
        == ALREADY_DECIDED
    )
    events = [line["event"] for line in audit.lines()]
    assert events == [
        "mention_received",
        "reply_sent",
        "action_received",
        "unauthorized_rejected",
        "action_received",
        "decision_recorded",
        "action_received",
        "duplicate_rejected",
    ]
    assert audit.lines()[0]["text"] == "<@UBOT> status"
    assert audit.lines()[1]["text"] == "No open cases."


def test_reject_submit_records_the_decision(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "slack.jsonl")
    log = DecisionLog(tmp_path / "decisions")
    metadata = json.dumps({"patch_id": "p9", "thread_id": "t9"})
    assert (
        on_reject_submit(
            user_id="UOWNER",
            comment="no",
            metadata=metadata,
            approvers=frozenset({"UOWNER"}),
            log=log,
            resume=lambda _thread, _payload: None,
            now=_NOW,
            audit=audit,
        )
        == ""
    )
    assert [line["event"] for line in audit.lines()] == ["action_received", "decision_recorded"]


def test_escalate_and_rerun_upload_no_diff_and_a_repair_matches_the_hash(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "slack.jsonl")
    gateway = RecordingGateway(MessageLog(tmp_path / "messages"))
    ports = ApprovalPorts(
        gateway=gateway,
        apply_root=tmp_path / "applied",
        postmortem_root=tmp_path / "postmortems",
        channel="C1",
        open_index=OpenIndex(tmp_path / "open.json"),
        audit=audit,
    )
    incident = make_incident()
    for kind in (RemediationKind.ESCALATE, RemediationKind.RERUN_UPSTREAM):
        patch = make_patch(incident_id=incident.id).model_copy(update={"remediation_kind": kind})
        case = IncidentCase(incident=incident, patch=patch, needs_human=True)
        ts = _post(ports, key=str(patch.id), text="Escalation", blocks=[], kind="escalation")
        _upload_diff(ports, case, str(patch.id), ts)
    assert gateway.uploads == []
    repair = make_patch(incident_id=incident.id).model_copy(
        update={"remediation_kind": RemediationKind.DATA_REPAIR, "diff": DIFF}
    )
    case = IncidentCase(incident=incident, patch=repair)
    ts = _post(ports, key=str(repair.id), text="Approval", blocks=[], kind="approval")
    _upload_diff(ports, case, str(repair.id), ts)
    _upload_diff(ports, case, str(repair.id), ts)
    assert gateway.uploads[0]["content"] == DIFF
    raw = DIFF.encode("utf-8")
    uploaded = [line for line in audit.lines() if line["event"] == "file_uploaded"]
    assert len(uploaded) == 1
    assert uploaded[0]["bytes"] == len(raw)
    assert uploaded[0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert uploaded[0]["name"] == f"{incident.id}.diff"
    _show(ports, str(repair.id), "Decision", [])
    updated = [line for line in audit.lines() if line["event"] == "message_updated"]
    assert updated[0]["buttons_removed"] is True
    assert updated[0]["ts"] == ts


def test_approver_mentions_survive_mrkdwn_escaping() -> None:
    assert mention_user("U0CAQ22LVL1") == "<@U0CAQ22LVL1>"
    assert mention_user("timeout") == "timeout"
    incident = make_incident()
    patch = make_patch(incident_id=incident.id)
    approval = make_approval(patch_id=patch.id).model_copy(update={"approver": "UOWNER"})
    case = IncidentCase(
        incident=incident,
        approval=approval,
        patch=patch,
        needs_human=True,
    )
    from meridian.slack.blocks import case_view

    blocks = decided_blocks(case_view(case), approval)
    section = blocks[-1]["text"]
    assert isinstance(section, dict)
    assert "<@UOWNER>" in str(section["text"])
    assert "&lt;@UOWNER&gt;" not in str(section["text"])


def test_open_index_keeps_the_first_posted_at(tmp_path: Path) -> None:
    index = OpenIndex(tmp_path / "open.json")
    first = _NOW
    later = _NOW + timedelta(hours=2)
    case = OpenCase(
        incident_id="i1",
        severity="SEV1",
        node="slack_approval",
        needs_human=True,
        detected_at=_NOW,
        posted_at=first,
        thread_id="t",
        patch_id="p",
    )
    index.upsert(case)
    index.upsert(
        OpenCase(
            incident_id="i1",
            severity="SEV2",
            node="slack_approval",
            needs_human=False,
            detected_at=_NOW,
            posted_at=later,
            thread_id="t",
            patch_id="p",
        )
    )
    stored = index.cases()[0]
    assert stored.posted_at == first
    assert stored.severity == "SEV2"
