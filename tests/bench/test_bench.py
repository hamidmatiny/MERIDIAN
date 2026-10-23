"""Splits, hard-mode frames, the final-run lock, and the report."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml
from tests.contracts.builders import make_incident

from meridian.agents.fix import active_fix_version
from meridian.bench.hard import hard_frame, within_sla_late
from meridian.bench.harness import _live_chats, execute_spec, guard_final, run_split
from meridian.bench.record import summarize_run
from meridian.bench.report import build_report, render_readme_results
from meridian.bench.splits import load_splits
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.partition_ref import PartitionRef
from meridian.contracts.state import MeridianState
from meridian.llm.errors import RunAborted
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE
from meridian.quality.contracts import MODELS
from meridian.quality.freshness import deadline, is_late, on_time_arrival
from meridian.run import PreparedPartition
from meridian.slack.gate import ApprovalPorts, run_approval
from meridian.slack.gateway import RecordingGateway
from meridian.slack.store import MessageLog, OpenIndex


def test_frozen_splits_have_the_promised_counts_and_disjoint_seeds() -> None:
    splits = load_splits()
    assert len(splits["dev"]) == 6
    assert len(splits["test"]) == 72
    assert len(splits["hard"]) == 30
    dev_seeds = {item.seed for item in splits["dev"]}
    test_seeds = {item.seed for item in splits["test"]}
    assert dev_seeds.isdisjoint(test_seeds)
    assert {item.fault for item in splits["hard"]} == {None}


def test_overlapping_seeds_are_refused(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("bench/splits.yaml").read_text(encoding="utf-8"))
    source["test"]["seeds"] = [101]
    path = tmp_path / "splits.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_splits(path)


def test_the_test_split_requires_final_and_a_second_final_requires_force(tmp_path: Path) -> None:
    info = {
        "prompt_versions": {"fix": "v1"},
        "commit": "abc",
        "dirty": False,
        "diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    with pytest.raises(SystemExit, match="requires --final"):
        guard_final(tmp_path, info, split="test", final=False, force=False)
    assert guard_final(tmp_path, info, split="test", final=True, force=False) is False
    assert guard_final(tmp_path, info, split="test", final=True, force=False) is False
    (tmp_path / "REPORT.md").write_text("done\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="--force"):
        guard_final(tmp_path, info, split="test", final=True, force=False)
    assert guard_final(tmp_path, info, split="test", final=True, force=True) is True
    lock = json.loads((tmp_path / "LOCK.json").read_text(encoding="utf-8"))
    assert lock["forced"] is True
    assert lock["commit"] == "abc"
    assert lock["dirty"] is False
    assert lock["diff_sha256"].startswith("e3b0c442")


def test_worktree_records_the_commit_the_dirty_flag_and_the_diff_hash() -> None:
    from meridian.bench.harness import _worktree

    info = _worktree()
    assert len(info["commit"]) >= 7
    assert isinstance(info["dirty"], bool)
    assert len(info["diff_sha256"]) == 64


def test_a_final_run_refuses_a_dirty_tree(tmp_path: Path) -> None:
    info = {"prompt_versions": {"fix": "v2"}, "commit": "abc", "dirty": True, "diff_sha256": "abc"}
    with pytest.raises(SystemExit, match="dirty tree"):
        guard_final(tmp_path, info, split="test", final=True, force=True)
    assert not (tmp_path / "LOCK.json").exists()


def test_hard_frames_stay_inside_the_contract_and_arrive_inside_the_sla() -> None:
    arrived = within_sla_late(DEFAULT_PARTITION_DATE)
    assert arrived < deadline(DEFAULT_PARTITION_DATE)
    assert arrived > on_time_arrival(DEFAULT_PARTITION_DATE)
    assert not is_late(
        type("Meta", (), {"arrived_at": arrived, "partition_date": DEFAULT_PARTITION_DATE})()
    )
    for dataset in ("vehicle_telemetry", "trip_summary", "fleet_daily_kpi"):
        frame = hard_frame(dataset, seed=301, rows=40, partition_date=DEFAULT_PARTITION_DATE)
        MODELS[dataset].validate(frame)
        assert not frame.isna().any().any()


def test_report_states_false_fix_as_n_over_n() -> None:
    run = {
        "fault": "NULL_EXPLOSION",
        "dataset": "vehicle_telemetry",
        "incidents": [{"fault": "NULL_EXPLOSION"}],
        "cases": [
            {
                "severity": "SEV2",
                "floor_raised": True,
                "has_hypothesis": True,
                "raw_causal_correct": True,
                "causal_event_correct": False,
                "columns_jaccard": 0.5,
                "grounding_violations": 1,
                "guardrail_fn": True,
                "kind_appropriate": True,
                "verified": True,
                "repaired_matches_clean": False,
                "escalated": False,
                "has_verdict": True,
                "critic_approved": True,
                "critic_degraded": False,
                "critic_corrective": True,
            }
        ],
        "machine_time_s": 12.0,
        "wall_s": 20.0,
        "input_tokens": 10,
        "output_tokens": 2,
        "usd": 0.0,
        "timeouts": 0,
        "wall_abort": False,
        "structured": {"fix": {"ok": 1, "n": 1}},
        "node_latency_ms": {"fix": [100]},
    }
    text, summary = build_report(
        [run],
        meta={
            "split": "dev",
            "commit": "abc",
            "models": {},
            "prompt_versions": {},
            "ollama": "test",
            "seeds": [101],
            "forced_final": False,
        },
    )
    assert summary["fix"]["false_fix"]["text"] == "1/1"
    scored = build_report(
        [run],
        meta={
            "split": "test",
            "commit": "abc",
            "models": {},
            "prompt_versions": {},
            "ollama": "test",
            "seeds": [201],
            "forced_final": False,
            "scorer_sha256": "abc123",
            "scorer_parent_commit": "7edbd17",
            "started_dirty": True,
            "mixed_version": False,
            "hard_mode_fp": {"n": 0, "N": 30, "text": "0/30"},
        },
    )
    assert "scorer version `abc123`" in scored[0]
    assert "started from a dirty tree" in scored[0]
    assert "mixed-version" not in scored[0]
    readme = render_readme_results(scored[1])
    assert "0/30" in readme
    assert "started from a dirty tree" in readme
    assert "abc123" in readme
    assert "1/1" in text
    assert summary["critic"]["matrix"]["approved_bad"] == 1
    assert "No ground-truth severity" in text


def test_bench_gate_does_not_apply(tmp_path: Path) -> None:
    gateway = RecordingGateway(MessageLog(tmp_path / "messages"))
    ports = ApprovalPorts(
        gateway=gateway,
        apply_root=tmp_path / "applied",
        postmortem_root=tmp_path / "postmortems",
        channel="bench",
        open_index=OpenIndex(tmp_path / "open.json"),
        bench=True,
    )
    incident = make_incident()
    state = MeridianState(run_id=uuid4(), cases={str(incident.id): IncidentCase(incident=incident)})
    assert run_approval(state, ports) == {}
    assert gateway.posts == []
    assert gateway.uploads == []
    assert not (tmp_path / "applied").exists()


def test_summarize_records_bench_no_human() -> None:
    incident = make_incident()
    state = MeridianState(run_id=uuid4(), cases={str(incident.id): IncidentCase(incident=incident)})
    frame = hard_frame("vehicle_telemetry", 1, 4, DEFAULT_PARTITION_DATE)
    record = summarize_run(
        state,
        split="hard",
        dataset="vehicle_telemetry",
        seed=301,
        fault=None,
        intensity=None,
        clean=frame,
        label=None,
        wall_s=1.0,
        wall_abort=False,
        run_key="k",
    )
    assert record["decision"] == "BENCH_NO_HUMAN"
    assert record["applied"] is False


def test_fix_version_defaults_to_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERIDIAN_FIX_PROMPT_VERSION", raising=False)
    assert active_fix_version() == "v1"
    monkeypatch.setenv("MERIDIAN_FIX_PROMPT_VERSION", "v2")
    assert active_fix_version() == "v2"


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        triage_model="qwen",
        rootcause_model="qwen",
        fix_model="qwen",
        critic_model="gemma",
        run_wall_budget_s=30,
        ray_num_cpus=1,
    )


def test_run_split_resumes_completed_keys_and_can_run_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("meridian.bench.harness.get_settings", _settings)
    monkeypatch.setattr("meridian.bench.harness._live_chats", lambda _settings: object())
    monkeypatch.setattr("meridian.bench.harness.parallel_workers", lambda: 1)
    calls: list[str] = []

    def fake_execute(spec: object, _settings: object, _chats: object, *, rows: int = 10) -> dict:
        del rows
        key = spec.key  # type: ignore[attr-defined]
        calls.append(key)
        return {
            "run_key": key,
            "split": "dev",
            "dataset": spec.dataset,  # type: ignore[attr-defined]
            "seed": spec.seed,  # type: ignore[attr-defined]
            "fault": spec.fault,  # type: ignore[attr-defined]
            "intensity": 0.1,
            "decision": "BENCH_NO_HUMAN",
            "applied": False,
            "completed": True,
            "wall_abort": False,
            "wall_s": 1.0,
            "machine_time_s": None,
            "incidents": [],
            "cases": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "usd": 0.0,
            "timeouts": 0,
            "structured": {},
            "node_latency_ms": {},
        }

    monkeypatch.setattr("meridian.bench.harness.execute_spec", fake_execute)
    directory = run_split("dev", final=False, force=False, resume=False)
    assert len(calls) == 6
    assert (directory / "REPORT.md").is_file()
    assert (directory / "summary.json").is_file()
    run_split("dev", final=False, force=False, resume=True)
    assert len(calls) == 6
    first = calls[0]
    run_split("dev", final=False, force=False, resume=False, only_key=first)
    assert calls[-1] == first


def test_a_crashed_run_is_left_for_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("meridian.bench.harness.get_settings", _settings)
    monkeypatch.setattr("meridian.bench.harness._live_chats", lambda _settings: object())
    monkeypatch.setattr("meridian.bench.harness.parallel_workers", lambda: 1)
    one = load_splits()["dev"][0]
    monkeypatch.setattr("meridian.bench.harness.load_splits", lambda: {"dev": [one]})

    def explode(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise TimeoutError("MCP stdio session did not open")

    monkeypatch.setattr("meridian.bench.harness.execute_spec", explode)
    directory = run_split(
        "dev", final=False, force=False, resume=False, log_path=tmp_path / "bench.log"
    )
    assert not (directory / "REPORT.md").is_file()
    assert "bench incomplete" in (tmp_path / "bench.log").read_text(encoding="utf-8")


def test_parallel_children_are_separate_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("meridian.bench.harness.get_settings", _settings)
    monkeypatch.setattr("meridian.bench.harness.parallel_workers", lambda: 2)
    launched: list[str] = []
    real_run = __import__("subprocess").run

    def fake_run(command: object, **kwargs: object) -> object:
        if not isinstance(command, list) or "--only-key" not in command:
            return real_run(command, **kwargs)  # type: ignore[arg-type]
        key = command[command.index("--only-key") + 1]
        launched.append(key)
        folder = tmp_path / "bench" / "results" / "hard"
        folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_key": key,
            "fault": "NULL_EXPLOSION",
            "dataset": "vehicle_telemetry",
            "completed": True,
            "incidents": [],
            "cases": [],
            "wall_s": 1,
            "usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "timeouts": 0,
            "wall_abort": False,
            "structured": {},
            "node_latency_ms": {},
        }
        (folder / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
        return None

    monkeypatch.setattr("meridian.bench.harness.subprocess.run", fake_run)
    directory = run_split("hard", final=False, force=False, resume=False)
    assert len(launched) == 30
    assert (directory / "summary.json").is_file()


def _prepared() -> PreparedPartition:
    frame = hard_frame("vehicle_telemetry", 1, 4, DEFAULT_PARTITION_DATE)
    return PreparedPartition(
        ref=PartitionRef(
            dataset="vehicle_telemetry",
            partition_date=DEFAULT_PARTITION_DATE,
            parquet_path="part.parquet",
            metadata=PartitionMetadata(
                dataset="vehicle_telemetry",
                partition_date=DEFAULT_PARTITION_DATE,
                arrived_at=datetime(2026, 10, 20, 1, 0, tzinfo=UTC),
            ),
            reference_profile_path="profile.json",
        ),
        clean=frame,
        label=None,
    )


def test_execute_spec_records_a_wall_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_splits()["dev"][0]

    def explode(*_args: object, **_kwargs: object) -> object:
        raise RunAborted("WALL_BUDGET", "budget spent")

    monkeypatch.setattr("meridian.bench.harness._prepare", lambda *_a, **_k: _prepared())
    monkeypatch.setattr("meridian.bench.harness._invoke", explode)
    record = execute_spec(spec, _settings(), object(), rows=4)  # type: ignore[arg-type]
    assert record["wall_abort"] is True
    assert record["decision"] == "BENCH_NO_HUMAN"


def test_execute_spec_summarizes_a_finished_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_splits()["dev"][0]
    incident = make_incident()
    state = MeridianState(run_id=uuid4(), cases={str(incident.id): IncidentCase(incident=incident)})
    monkeypatch.setattr("meridian.bench.harness._prepare", lambda *_a, **_k: _prepared())
    monkeypatch.setattr("meridian.bench.harness._invoke", lambda *_a, **_k: state)
    record = execute_spec(spec, _settings(), object(), rows=4)  # type: ignore[arg-type]
    assert record["completed"] is True
    assert record["applied"] is False
    assert record["incidents"][0]["fault"] == incident.fault_type.value


def test_live_chats_stops_when_the_budget_or_a_method_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("meridian.bench.harness._chat", lambda *_a, **_k: "chats")
    monkeypatch.setattr("meridian.bench.harness.read_ollama_sizes", dict)
    monkeypatch.setattr("meridian.bench.harness.size_report", lambda *_a, **_k: "sizes")
    monkeypatch.setattr("meridian.bench.harness.budget_problem", lambda *_a, **_k: "too big")
    with pytest.raises(SystemExit, match="too big"):
        _live_chats(_settings())  # type: ignore[arg-type]
    monkeypatch.setattr("meridian.bench.harness.budget_problem", lambda *_a, **_k: None)
    monkeypatch.setattr("meridian.bench.harness._capabilities", lambda *_a, **_k: "no method")
    with pytest.raises(SystemExit, match="no method"):
        _live_chats(_settings())  # type: ignore[arg-type]
    monkeypatch.setattr("meridian.bench.harness._capabilities", lambda *_a, **_k: None)
    assert _live_chats(_settings()) == "chats"  # type: ignore[arg-type]


def test_is_late_uses_a_real_metadata_object() -> None:
    """The hard arrival is inside the SLA by more than a minute."""
    assert within_sla_late(DEFAULT_PARTITION_DATE) + timedelta(seconds=60) == deadline(
        DEFAULT_PARTITION_DATE
    )
