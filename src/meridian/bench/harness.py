"""Run one bench split. Sequential unless MERIDIAN_BENCH_PARALLEL is raised.

Ollama serves one request at a time unless OLLAMA_NUM_PARALLEL is at least the bench
parallelism. The default is 1 because the two resident models already fill the memory cap.
A higher bench parallel queues on the GPU when OLLAMA_NUM_PARALLEL is left at 1.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from meridian.agents.fix import active_fix_version
from meridian.bench.hard import prepare_hard
from meridian.bench.record import summarize_run
from meridian.bench.report import build_report
from meridian.bench.splits import RunSpec, load_splits
from meridian.config import Settings, get_settings
from meridian.llm.errors import AbortGate, RunAborted
from meridian.llm.resident import budget_problem, read_ollama_sizes, size_report
from meridian.pipeline.catalog import DEFAULT_ROWS
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE
from meridian.prompts import (
    CRITIC_PROMPT_VERSION,
    FIX_PROMPT_VERSION,
    ROOT_CAUSE_PROMPT_VERSION,
    TRIAGE_PROMPT_VERSION,
)
from meridian.run import _capabilities, _chat, _invoke, _prepare
from meridian.wall import WallClock

_LOG = Path("data/bench.log")


def results_root() -> Path:
    return Path("bench/results")


def bench_id(split: str) -> str:
    """One directory per split and fix-prompt version, so v1 and v2 do not collide."""
    if split == "hard":
        return "hard"
    return f"{split}-fix-{active_fix_version()}"


def parallel_workers() -> int:
    raw = os.environ.get("MERIDIAN_BENCH_PARALLEL", "1").strip() or "1"
    try:
        count = int(raw)
    except ValueError as exc:
        raise SystemExit("MERIDIAN_BENCH_PARALLEL must be an integer") from exc
    if count < 1:
        raise SystemExit("MERIDIAN_BENCH_PARALLEL must be >= 1")
    return count


def identity(settings: Settings) -> dict[str, Any]:
    """Lock fields include the commit, whether the tree is dirty, and the diff hash."""
    tree = _worktree()
    return {
        **tree,
        "prompt_versions": {
            "triage": TRIAGE_PROMPT_VERSION,
            "root_cause": ROOT_CAUSE_PROMPT_VERSION,
            "fix": active_fix_version(),
            "fix_default": FIX_PROMPT_VERSION,
            "critic": CRITIC_PROMPT_VERSION,
        },
        "models": {
            "triage": settings.triage_model,
            "root_cause": settings.rootcause_model,
            "fix": settings.fix_model,
            "critic": settings.critic_model,
        },
        "ollama": _ollama_version(),
    }


def guard_final(
    directory: Path,
    info: dict[str, Any],
    *,
    split: str,
    final: bool,
    force: bool,
) -> bool:
    """Refuse the test split without --final.

    A finished final (``REPORT.md`` is present) of the same prompts needs ``--force``.
    An unfinished directory is the same final and may be resumed.
    """
    if split == "test" and not final:
        raise SystemExit("the test split requires --final")
    if final and info.get("dirty"):
        raise SystemExit("a final run refuses a dirty tree")
    if not final:
        return False
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "LOCK.json"
    flagged = False
    finished = (directory / "REPORT.md").is_file()
    if lock.is_file() and finished:
        previous = json.loads(lock.read_text(encoding="utf-8"))
        same = previous.get("prompt_versions") == info["prompt_versions"]
        if same and not force:
            raise SystemExit("a final run for these prompt versions exists; pass --force")
        flagged = same and force
    payload = {**info, "forced": flagged}
    lock.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return flagged


def execute_spec(
    spec: RunSpec,
    settings: Settings,
    chats: Any,
    *,
    rows: int = DEFAULT_ROWS,
) -> dict[str, Any]:
    """One headless graph run. The gate records BENCH_NO_HUMAN and does not apply."""
    clock = WallClock(settings.run_wall_budget_s)
    run_id = uuid4()
    clock.start()
    try:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            if spec.fault is None:
                prepared = prepare_hard(
                    root,
                    dataset=spec.dataset,
                    seed=spec.seed,
                    rows=rows,
                    partition=DEFAULT_PARTITION_DATE,
                )
            else:
                from meridian.contracts.fault_type import FaultType

                if spec.intensity is None:
                    raise RuntimeError("injected run is missing intensity")
                prepared = _prepare(
                    root,
                    dataset=spec.dataset,
                    fault=FaultType(spec.fault),
                    intensity=spec.intensity,
                    seed=spec.seed,
                    rows=rows,
                    partition=DEFAULT_PARTITION_DATE,
                )
            _isolate_run(chats, clock)
            try:
                state = _invoke(
                    settings,
                    chats,
                    prepared.ref,
                    root,
                    bench=True,
                    clock=clock,
                    run_id=run_id,
                )
            except RunAborted:
                return _aborted(spec, clock.elapsed())
            if state is None:
                raise RuntimeError("bench reached a Slack interrupt")
            from pandas import DataFrame

            if not isinstance(prepared.clean, DataFrame):
                raise RuntimeError("clean frame missing")
            return summarize_run(
                state,
                split=spec.split,
                dataset=spec.dataset,
                seed=spec.seed,
                fault=spec.fault,
                intensity=spec.intensity,
                clean=prepared.clean,
                label=prepared.label,
                wall_s=clock.elapsed(),
                wall_abort=False,
                run_key=spec.key,
            )
    finally:
        clock.stop()


def run_split(
    split: str,
    *,
    final: bool,
    force: bool,
    resume: bool,
    only_key: str | None = None,
    log_path: Path = _LOG,
) -> Path:
    """Execute the split and write REPORT.md. Returns the results directory."""
    specs = load_splits()[split]
    if only_key is not None:
        specs = [spec for spec in specs if spec.key == only_key]
        if not specs:
            raise SystemExit(f"unknown run key {only_key}")
    settings = get_settings()
    info = identity(settings)
    directory = results_root() / bench_id(split)
    flagged = (
        False if only_key else guard_final(directory, info, split=split, final=final, force=force)
    )
    directory.mkdir(parents=True, exist_ok=True)
    pending = [spec for spec in specs if not (resume and _done(directory / f"{spec.key}.json"))]
    _log(log_path, f"bench split={split} id={directory.name} pending={len(pending)}/{len(specs)}")
    _log(
        log_path,
        "OLLAMA_NUM_PARALLEL should be >= MERIDIAN_BENCH_PARALLEL "
        f"({parallel_workers()}); otherwise Ollama queues the extra requests",
    )
    if pending and only_key is None and parallel_workers() > 1:
        _run_parallel(split, pending, directory, final, force, resume, log_path)
    else:
        chats = _live_chats(settings)
        started = time.monotonic()
        finished = len(specs) - len(pending)
        for spec in pending:
            record = _execute_resilient(spec, settings, chats, log_path)
            if record is None:
                continue
            _write_record(directory, record)
            finished += 1
            _progress(log_path, finished, len(specs), started)
    if only_key is None and all(_done(directory / f"{spec.key}.json") for spec in specs):
        _write_report(directory, split, info, flagged, specs)
    elif only_key is None:
        _log(log_path, "bench incomplete; rerun with --resume")
    return directory


def _execute_resilient(
    spec: RunSpec,
    settings: Settings,
    chats: Any,
    log_path: Path,
) -> dict[str, Any] | None:
    """Retry a crashed run once. A second failure is left for ``--resume``."""
    last = "unknown"
    for attempt in (1, 2):
        try:
            return execute_spec(spec, settings, chats)
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            _log(log_path, f"run {spec.key} attempt {attempt} failed: {last}")
    return None


def _run_parallel(
    split: str,
    pending: list[RunSpec],
    directory: Path,
    final: bool,
    force: bool,
    resume: bool,
    log_path: Path,
) -> None:
    """Child processes, so each run has its own Ray cluster. The GPU still queues."""
    from concurrent.futures import ThreadPoolExecutor

    def launch(spec: RunSpec) -> None:
        command = [
            sys.executable,
            "-m",
            "meridian.bench",
            "--split",
            split,
            "--only-key",
            spec.key,
        ]
        if final:
            command.append("--final")
        if force:
            command.append("--force")
        if resume:
            command.append("--resume")
        subprocess.run(command, check=True)

    started = time.monotonic()
    workers = parallel_workers()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(launch, pending))
    _progress(log_path, len(pending), len(pending), started)


def _live_chats(settings: Settings) -> Any:
    clock = WallClock(settings.run_wall_budget_s)
    clock.start()
    try:
        chats = _chat(settings, live=True, clock=clock)
        sizes = read_ollama_sizes()
        _log(_LOG, size_report(settings, sizes))
        problem = budget_problem(settings, sizes)
        if problem is not None:
            raise SystemExit(problem)
        blocked = _capabilities(settings, chats)
        if blocked is not None:
            raise SystemExit(blocked)
        return chats
    finally:
        clock.stop()


def _write_report(
    directory: Path,
    split: str,
    info: dict[str, Any],
    flagged: bool,
    specs: list[RunSpec],
) -> None:
    runs = []
    for spec in specs:
        path = directory / f"{spec.key}.json"
        if path.is_file():
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    comparisons = _comparisons(split)
    meta = {
        **info,
        "split": split,
        "seeds": sorted({spec.seed for spec in specs}),
        "forced_final": flagged,
    }
    markdown, summary = build_report(runs, meta=meta, comparisons=comparisons)
    (directory / "REPORT.md").write_text(markdown, encoding="utf-8")
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _comparisons(split: str) -> list[tuple[str, dict[str, Any]]]:
    if split != "test":
        return []
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(results_root().glob("dev-fix-*/summary.json")):
        found.append((path.parent.name, json.loads(path.read_text(encoding="utf-8"))))
    return found


def _isolate_run(chats: Any, clock: WallClock) -> None:
    """Give this run a new abort gate, clock, and deadline.

    The capability probe builds one chat and one gate for the whole process.
    Leaving that gate in place made a later run inherit an earlier abort, and the
    probe deadline expired after the first 900 seconds of the split.
    """
    gate = AbortGate()
    deadline = clock.started + clock.budget_s
    for name in ("triage_chat", "root_chat", "fix_chat", "critic_chat"):
        chat = getattr(chats, name, None)
        if chat is None or not hasattr(chat, "gate"):
            continue
        chat.gate = gate
        chat.clock = clock
        chat.deadline = deadline


def _aborted(spec: RunSpec, wall_s: float) -> dict[str, Any]:
    return {
        "run_key": spec.key,
        "split": spec.split,
        "dataset": spec.dataset,
        "seed": spec.seed,
        "fault": spec.fault,
        "intensity": spec.intensity,
        "decision": "BENCH_NO_HUMAN",
        "applied": False,
        "completed": True,
        "wall_abort": True,
        "abort_reason": "WALL_BUDGET",
        "wall_s": wall_s,
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


def _write_record(directory: Path, record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['run_key']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _done(path: Path) -> bool:
    if not path.is_file():
        return False
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return bool(isinstance(loaded, dict) and loaded.get("completed"))


def _progress(log_path: Path, done: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = elapsed / done if done else 0.0
    eta = rate * (total - done)
    _log(log_path, f"progress {done}/{total} eta_s={eta:.0f}")


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = message.rstrip()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, file=sys.stderr)


def _worktree() -> dict[str, Any]:
    """``git diff HEAD`` hashed at start. A dirty tree cannot begin ``--final``."""
    diff = _git_text(["git", "diff", "HEAD"])
    porcelain = _git_text(["git", "status", "--porcelain"])
    return {
        "commit": _git_head(),
        "dirty": bool(porcelain.strip()),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _git_head() -> str:
    return _git_text(["git", "rev-parse", "HEAD"]).strip() or "unknown"


def _git_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True)
    except (OSError, subprocess.CalledProcessError):
        return ""


def _ollama_version() -> str:
    try:
        return subprocess.check_output(["ollama", "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
