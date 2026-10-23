"""Branches the live bench never hits in the offline suite."""

import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
from tests.bench.test_bench import _prepared, _settings
from tests.contracts.builders import (
    make_hypothesis,
    make_incident,
    make_patch,
    make_verdict,
    make_verification,
)

from meridian.agents.fix import _feedback
from meridian.bench.__main__ import main
from meridian.bench.hard import _perturb, _retie_trip_speed, prepare_hard
from meridian.bench.harness import (
    _git_head,
    _ollama_version,
    _run_parallel,
    execute_spec,
    parallel_workers,
    run_split,
)
from meridian.bench.record import _case_row, summarize_run
from meridian.bench.report import build_report
from meridian.bench.splits import RunSpec, load_splits
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.injected_fault import InjectedFault
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.severity import Severity
from meridian.contracts.state import MeridianState
from meridian.contracts.triage import TriageResult
from meridian.progress import node_line
from meridian.prompts import load_prompt

_LABEL = UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a11")


def test_cli_prints_the_results_directory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "meridian.bench.__main__.run_split", lambda *_a, **_k: Path("bench/results/hard")
    )
    assert main(["--split", "hard", "--resume"]) == 0
    assert capsys.readouterr().out.strip() == "bench/results/hard"


def test_prepare_hard_writes_every_dataset(tmp_path: Path) -> None:
    for dataset in ("vehicle_telemetry", "trip_summary", "fleet_daily_kpi"):
        folder = tmp_path / dataset
        folder.mkdir()
        prepared = prepare_hard(
            folder,
            dataset=dataset,
            seed=1,
            rows=4,
            partition=date(2026, 10, 19),
        )
        assert prepared.label is None
        assert Path(prepared.ref.parquet_path).is_file()


def test_noise_hits_tails_bounds_and_a_too_long_trip() -> None:
    short = _perturb(np.zeros(1), 0.0, 10.0, np.random.default_rng(1))
    assert len(short) == 1
    long = _perturb(np.zeros(800), 0.0, 10.0, np.random.default_rng(2))
    assert long[0] == 0.0
    assert long[-1] == 10.0
    start = pd.Timestamp("2026-01-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "start_ts": [start],
            "end_ts": [start + pd.Timedelta(hours=10)],
            "avg_speed_mps": [60.0],
            "distance_km": [1.0],
        }
    )
    _retie_trip_speed(frame)
    assert float(frame["distance_km"].iloc[0]) == 500.0


def test_split_yaml_rejects_bad_shapes(tmp_path: Path) -> None:
    broken = tmp_path / "splits.yaml"
    broken.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_splits(broken)
    broken.write_text(
        "dev: []\ntest: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [0.1], seeds: [2]}\n"
        "hard: {datasets: [a], seeds: [3]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dev split"):
        load_splits(broken)
    broken.write_text(
        """
dev: {faults: [NOPE], datasets: [a], intensities: [0.1], seeds: [1]}
test: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [0.1], seeds: [2]}
hard: {datasets: [a], seeds: [3]}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fault"):
        load_splits(broken)
    broken.write_text(
        """
dev: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [0.1], seeds: [9]}
test: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [0.1], seeds: [9]}
hard: {datasets: [a], seeds: [3]}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_splits(broken)
    broken.write_text(
        """
dev: {faults: [], datasets: [a], intensities: [0.1], seeds: [1]}
test: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [0.1], seeds: [2]}
hard: {datasets: [a], seeds: [3]}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="faults"):
        load_splits(broken)
    broken.write_text(
        """
dev: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [true], seeds: [1]}
test: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [0.1], seeds: [2]}
hard: {datasets: [a], seeds: ["x"]}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="intensities"):
        load_splits(broken)
    broken.write_text(
        """
dev: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [0.1], seeds: [1]}
test: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [], seeds: [2]}
hard: {datasets: [a], seeds: [3]}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="intensities"):
        load_splits(broken)
    broken.write_text(
        """
dev: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [0.1], seeds: [1]}
test: {faults: [NULL_EXPLOSION], datasets: [a], intensities: [0.1], seeds: [2]}
hard: {datasets: [a], seeds: ["x"]}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seeds"):
        load_splits(broken)


def test_parallel_setting_and_missing_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERIDIAN_BENCH_PARALLEL", "no")
    with pytest.raises(SystemExit, match="integer"):
        parallel_workers()
    monkeypatch.setenv("MERIDIAN_BENCH_PARALLEL", "0")
    with pytest.raises(SystemExit, match=">= 1"):
        parallel_workers()
    monkeypatch.setenv("MERIDIAN_BENCH_PARALLEL", "")
    assert parallel_workers() == 1

    def boom(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(1, "tool")

    monkeypatch.setattr("meridian.bench.harness.subprocess.check_output", boom)
    assert _git_head() == "unknown"
    assert _ollama_version() == "unknown"


def test_each_run_gets_a_fresh_abort_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from meridian.llm.errors import AbortGate

    spent = AbortGate()
    spent.trip("WALL_BUDGET: run wall budget exceeded")
    chat = SimpleNamespace(gate=spent, clock=None, deadline=0.0)
    chats = SimpleNamespace(
        triage_chat=chat,
        root_chat=chat,
        fix_chat=chat,
        critic_chat=chat,
    )
    spec = load_splits()["dev"][0]
    monkeypatch.setattr("meridian.bench.harness._prepare", lambda *_a, **_k: _prepared())
    monkeypatch.setattr("meridian.bench.harness._invoke", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="Slack interrupt"):
        execute_spec(spec, _settings(), chats, rows=4)
    assert chat.gate.reason is None
    assert chat.clock is not None
    assert chat.deadline > chat.clock.started


def test_execute_spec_hard_and_bad_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_splits()["hard"][0]
    monkeypatch.setattr("meridian.bench.harness._invoke", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="Slack interrupt"):
        execute_spec(spec, _settings(), object(), rows=4)  # type: ignore[arg-type]
    missing = RunSpec("dev", "vehicle_telemetry", 101, "NULL_EXPLOSION", None)
    with pytest.raises(RuntimeError, match="intensity"):
        execute_spec(missing, _settings(), object(), rows=4)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "meridian.bench.harness._prepare",
        lambda *_a, **_k: SimpleNamespace(ref=object(), clean="nope", label=None),
    )
    monkeypatch.setattr("meridian.bench.harness._invoke", lambda *_a, **_k: object())
    present = load_splits()["dev"][0]
    with pytest.raises(RuntimeError, match="clean frame"):
        execute_spec(present, _settings(), object(), rows=4)  # type: ignore[arg-type]


def test_unknown_key_and_parallel_flags_and_test_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="unknown run key"):
        run_split("dev", final=False, force=False, resume=False, only_key="missing")
    spec = load_splits()["hard"][0]
    seen: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        del check
        seen.append(command)

    monkeypatch.setattr("meridian.bench.harness.subprocess.run", fake_run)
    monkeypatch.setattr("meridian.bench.harness.parallel_workers", lambda: 1)
    _run_parallel("hard", [spec], tmp_path, True, True, True, tmp_path / "bench.log")
    assert "--final" in seen[0]
    assert "--force" in seen[0]
    assert "--resume" in seen[0]

    monkeypatch.chdir(tmp_path)
    one = RunSpec("test", "vehicle_telemetry", 201, "NULL_EXPLOSION", 0.1)
    monkeypatch.setattr("meridian.bench.harness.load_splits", lambda: {"test": [one]})
    monkeypatch.setattr("meridian.bench.harness.get_settings", _settings)
    monkeypatch.setattr(
        "meridian.bench.harness.identity",
        lambda _settings: {
            "commit": "abc",
            "prompt_versions": {"fix": "v1"},
            "models": {},
            "ollama": "test",
        },
    )
    monkeypatch.setattr("meridian.bench.harness._live_chats", lambda _settings: object())
    monkeypatch.setattr(
        "meridian.bench.harness.execute_spec",
        lambda *_a, **_k: {
            "run_key": one.key,
            "fault": "NULL_EXPLOSION",
            "dataset": "vehicle_telemetry",
            "completed": True,
            "incidents": [{"fault": "NULL_EXPLOSION"}],
            "cases": [],
            "wall_s": 1,
            "usd": 0,
            "input_tokens": 1,
            "output_tokens": 1,
            "timeouts": 0,
            "wall_abort": False,
            "structured": {},
            "node_latency_ms": {},
            "machine_time_s": 3,
        },
    )
    other = tmp_path / "bench" / "results" / "dev-fix-v1"
    other.mkdir(parents=True)
    (other / "summary.json").write_text("{}\n", encoding="utf-8")
    directory = run_split(
        "test", final=True, force=False, resume=False, log_path=tmp_path / "bench.log"
    )
    report = (directory / "REPORT.md").read_text(encoding="utf-8")
    assert "Dev and test" in report
    assert "dev-fix-v1" in report


def test_summarize_scores_tokens_and_machine_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MERIDIAN_PROGRESS_DIR", str(tmp_path))
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    verdict = make_verdict(patch_id=patch.id).model_copy(
        update={"approved": True, "critic_degraded": True, "corrective": True}
    )
    triage = TriageResult(
        incident_id=incident.id,
        severity=Severity.SEV2,
        rationale="Floor won.",
        affected_consumers=["trip_summary"],
        confidence=0.4,
        llm_severity=Severity.SEV3,
        floor_severity=Severity.SEV2,
        final_severity=Severity.SEV2,
        prompt_version="v1",
    )
    case = IncidentCase(
        incident=incident,
        triage=triage,
        hypothesis=hypothesis,
        patch=patch,
        verdicts=[verdict],
        verification=make_verification(patch_id=patch.id, passed=True),
        needs_human=True,
    )
    run_id = uuid4()
    node_line(run_id, str(incident.id), "slack_approval", 1, 1.0, "start")
    state = MeridianState(
        run_id=run_id,
        cases={str(incident.id): case},
        llm_calls=[
            LLMCall(
                node="fix",
                model="fake",
                prompt_version="v1",
                input_tokens=3,
                output_tokens=1,
                latency_ms=12,
                attempt=1,
                succeeded=True,
                structured=True,
            ),
            LLMCall(
                node="fix",
                model="fake",
                prompt_version="v1",
                input_tokens=1,
                output_tokens=0,
                latency_ms=4,
                attempt=2,
                succeeded=False,
                structured=True,
                timed_out=True,
            ),
        ],
        aborted=True,
        abort_reason="WALL_BUDGET",
    )

    class _Score:
        def model_dump(self) -> dict[str, object]:
            return {
                "raw_causal_correct": True,
                "causal_event_correct": False,
                "columns_jaccard": 0.5,
                "kind_appropriate": True,
                "repaired_matches_clean": False,
            }

    monkeypatch.setattr("meridian.bench.record.score_root_cause", lambda *_a, **_k: _Score())
    monkeypatch.setattr("meridian.bench.record.score_fix", lambda *_a, **_k: _Score())
    label = InjectedFault(
        id=_LABEL,
        fault_type=FaultType.NULL_EXPLOSION,
        dataset="vehicle_telemetry",
        columns_affected=["battery_pct"],
        rows_affected=1,
        intensity=0.1,
        seed=101,
        causal_event_id=uuid4(),
    )
    record = summarize_run(
        state,
        split="dev",
        dataset="vehicle_telemetry",
        seed=101,
        fault="NULL_EXPLOSION",
        intensity=0.1,
        clean=_prepared().clean,
        label=label,
        wall_s=2.0,
        wall_abort=False,
        run_key="one",
    )
    assert record["wall_abort"] is True
    assert record["machine_time_s"] is not None
    assert record["timeouts"] == 1
    assert record["structured"]["fix"] == {"ok": 1, "n": 2}
    row = record["cases"][0]
    assert row["guardrail_fn"] is True
    assert row["floor_raised"] is True
    assert row["critic_approved"] is True
    with pytest.raises(TypeError, match="IncidentCase"):
        _case_row(object(), _prepared().clean, None)
    markdown, summary = build_report(
        [record, {"fault": None, "incidents": [{"fault": "X"}], "cases": [], "wall_s": 1}],
        meta={"split": "test", "commit": "abc", "forced_final": False},
        comparisons=[("dev-fix-v1", {})],
    )
    assert "FALSE-FIX" in markdown or "false-fix" in markdown
    assert summary["hard_fp"]["incidents"] == 1
    assert summary["comparisons"][0]["label"] == "dev-fix-v1"


def test_v2_feedback_names_the_previous_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    incident = make_incident()
    case = IncidentCase(
        incident=incident,
        patch=make_patch(incident_id=incident.id),
        needs_human=True,
    )
    monkeypatch.delenv("MERIDIAN_FIX_PROMPT_VERSION", raising=False)
    assert "previous_kind" not in _feedback(case)
    monkeypatch.setenv("MERIDIAN_FIX_PROMPT_VERSION", "v2")
    text = _feedback(case)
    assert text.splitlines()[0] == "previous_kind: ESCALATE"
    assert "submit_patch fields:" in text
    assert "$feedback" in load_prompt("fix", "v2")
