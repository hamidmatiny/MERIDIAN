"""Re-score saved bench JSON with score_fix. No model calls.

A DATA_REPAIR or QUARANTINE match needs the patch. The result JSON does not
store it. The checkpoint written at the end of that run does. The join is the
incident id plus the result-file mtime. A run with no such checkpoint keeps
its other fields and marks repaired_matches_clean as n/a.
"""

import hashlib
import json
import sqlite3
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from meridian.agents.patch_text import unified_diff
from meridian.bench.report import build_report, render_readme_results
from meridian.bench.score_fix import _APPROPRIATE, score
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident import Incident
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.patch import Patch
from meridian.contracts.remediation import RemediationKind
from meridian.injector.faults import INJECTORS
from meridian.pipeline.catalog import DEFAULT_ROWS
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE, generate_dataset

_ROOT = Path(__file__).resolve().parents[1]
_DIRS = ("dev-fix-v1", "dev-fix-v2", "hard", "test-fix-v2")
_WINDOW = timedelta(seconds=180)


def main() -> None:
    warnings.filterwarnings("ignore")
    parent = _git_head()
    digest = hashlib.sha256((_ROOT / "src/meridian/bench/score_fix.py").read_bytes()).hexdigest()
    before = {name: _load_summary(name) for name in _DIRS}
    print("indexing checkpoints")
    candidates = _checkpoint_cases()
    print(f"checkpoint candidates {len(candidates)}")
    faults = _infra_faults()
    frames: dict[tuple[str, int], Any] = {}
    pending: list[tuple[Path, dict[str, Any]]] = []
    for name in _DIRS:
        for path in sorted((_ROOT / "bench/results" / name).glob("*.json")):
            if path.name in {"summary.json", "LOCK.json"}:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            _annotate_infra(record, faults)
            if record.get("fault"):
                print(record.get("run_key"), flush=True)
                _rescore_record(record, path, candidates, frames)
            pending.append((path, record))
    for path, record in pending:
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hard_fp = None
    for name in _DIRS:
        summary_path = _ROOT / "bench/results" / name / "summary.json"
        previous = before[name] or {}
        meta = dict(previous.get("meta") or {})
        meta["forced_final"] = bool(previous.get("forced_final"))
        meta["scorer_sha256"] = digest
        meta["scorer_parent_commit"] = parent
        meta["mixed_version"] = False
        meta["started_dirty"] = name == "test-fix-v2"
        if name == "test-fix-v2" and hard_fp is not None:
            meta["hard_mode_fp"] = hard_fp
        runs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((_ROOT / "bench/results" / name).glob("*.json"))
            if path.name not in {"summary.json", "LOCK.json"}
        ]
        comparisons = _dev_comparisons() if name == "test-fix-v2" else None
        text, summary = build_report(runs, meta=meta, comparisons=comparisons)
        body = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        summary_path.write_text(body, encoding="utf-8")
        (_ROOT / "bench/results" / name / "REPORT.md").write_text(text, encoding="utf-8")
        if name == "hard":
            hard_fp = summary["hard_fp"]
        _print_delta(name, previous, summary)
    _write_readme()
    print("scorer", digest)
    print("parent", parent)


def _rescore_record(
    record: dict[str, Any],
    path: Path,
    candidates: list[dict[str, Any]],
    frames: dict[tuple[str, int], Any],
) -> None:
    fault = FaultType(record["fault"])
    key = (str(record["dataset"]), int(record["seed"]))
    if key not in frames:
        frames[key] = generate_dataset(
            key[0],
            key[1],
            rows=DEFAULT_ROWS,
            partition_date=DEFAULT_PARTITION_DATE,
        )
    clean = frames[key]
    _unused, label = INJECTORS[fault](
        clean,
        np.random.default_rng(int(record["seed"])),
        float(record["intensity"]),
        dataset=key[0],
        seed=int(record["seed"]),
    )
    incident_ids = {str(item["id"]) for item in record.get("incidents", [])}
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    for case in record.get("cases", []):
        kind = case.get("kind")
        if kind in {None, "ESCALATE", "RERUN_UPSTREAM"}:
            built = None if kind is None else _synthetic(record, label, RemediationKind(kind))
            _apply(case, built, clean, label, "saved kind does not need the patch diff")
            continue
        found, match = _closest(candidates, incident_ids, str(kind), mtime)
        if match is None:
            case["kind_appropriate"] = RemediationKind(kind) in _APPROPRIATE[fault]
            case["repaired_matches_clean"] = None
            case["rescore"] = "n/a"
            case["rescore_reason"] = found
            continue
        _apply(case, match["case"], clean, label, f"checkpoint {match['thread']}")


def _apply(
    case: dict[str, Any],
    built: IncidentCase | None,
    clean: Any,
    label: Any,
    reason: str,
) -> None:
    if built is None:
        case["kind_appropriate"] = False
        case["repaired_matches_clean"] = False
        case["rescore"] = "ok"
        case["rescore_reason"] = reason
        return
    try:
        scored = score(built, clean, label)
    except Exception as exc:
        case["repaired_matches_clean"] = None
        case["rescore"] = "n/a"
        case["rescore_reason"] = f"{type(exc).__name__}: {exc}"
        return
    case["kind_appropriate"] = scored.kind_appropriate
    case["repaired_matches_clean"] = scored.repaired_matches_clean
    case["rescore"] = "ok"
    case["rescore_reason"] = reason


def _closest(
    candidates: list[dict[str, Any]],
    incident_ids: set[str],
    kind: str,
    mtime: datetime,
) -> tuple[str, dict[str, Any] | None]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        if item["incident_id"] not in incident_ids or item["kind"] != kind:
            continue
        delta = (mtime - item["ts"]).total_seconds()
        if -30 <= delta <= _WINDOW.total_seconds():
            ranked.append((abs(delta), item))
    if not ranked:
        return (
            "saved artifact has no patch diff and no checkpoint within 180s "
            "shares this incident id and kind",
            None,
        )
    ranked.sort(key=lambda pair: pair[0])
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 15:
        return ("two checkpoints are within 15s of this result file", None)
    return ("ok", ranked[0][1])


def _synthetic(record: dict[str, Any], label: Any, kind: RemediationKind) -> IncidentCase:
    incident = Incident(
        id=uuid4(),
        detected_at=datetime(2026, 10, 20, tzinfo=UTC),
        dataset=str(record["dataset"]),
        partition="2026-10-19",
        contract_name="bench",
        fault_type=label.fault_type,
        failing_checks=["check"],
    )
    path = f"remediations/{incident.id}.py"
    patch = Patch(
        id=uuid4(),
        incident_id=incident.id,
        hypothesis_id=uuid4(),
        remediation_kind=kind,
        diff=unified_diff(path, "def repair(df):\n    return df\n"),
        regression_test="def check(df):\n    return None\n",
        files_touched=[path],
        author_model="rescore",
    )
    return IncidentCase(
        incident=incident,
        patch=patch,
        needs_human=kind is RemediationKind.ESCALATE,
    )


def _checkpoint_cases() -> list[dict[str, Any]]:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    database = _ROOT / "data/meridian-checkpoints.sqlite"
    connection = sqlite3.connect(database)
    serde = JsonPlusSerializer()
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    rows = connection.execute("SELECT thread_id, type, checkpoint FROM checkpoints")
    for thread, stored_type, blob in rows:
        data = serde.loads_typed((stored_type, blob))
        stamp = str(data.get("ts") or "")
        previous = latest.get(thread)
        if previous is not None and stamp <= previous[0]:
            continue
        channels = data.get("channel_values") or {}
        cases = channels.get("cases") or {}
        chosen = _patched_case(cases)
        if chosen is None:
            continue
        latest[thread] = (
            stamp,
            {
                "thread": thread,
                "ts": datetime.fromisoformat(stamp),
                "incident_id": _incident_id(chosen),
                "kind": _patch_kind(chosen),
                "case": chosen,
            },
        )
    return [item for _stamp, item in latest.values()]


def _patched_case(cases: object) -> IncidentCase | None:
    if not isinstance(cases, dict):
        return None
    for case in cases.values():
        if not isinstance(case, IncidentCase) or case.patch is None:
            continue
        if not isinstance(case.patch, Patch):
            continue
        if not _incident_id(case):
            continue
        return case
    return None


def _incident_id(case: IncidentCase) -> str:
    incident = case.incident
    if isinstance(incident, dict):
        return str(incident.get("id", ""))
    return str(getattr(incident, "id", ""))


def _patch_kind(case: IncidentCase) -> str:
    patch = case.patch
    if not isinstance(patch, Patch):
        return ""
    return patch.remediation_kind.value


def _infra_faults() -> dict[str, list[str]]:
    log = (_ROOT / "data/bench.log").read_text(encoding="utf-8", errors="replace")
    found: dict[str, list[str]] = {}
    for line in log.splitlines():
        if " attempt 1 failed:" not in line:
            continue
        key = line.split(" attempt 1 failed:", 1)[0].replace("run ", "", 1).strip()
        detail = line.split(" attempt 1 failed:", 1)[1].strip()
        name = detail.split(":", 1)[0].strip()
        faults = found.setdefault(key, [])
        label = f"attempt_1_{name}"
        if label not in faults:
            faults.append(label)
    head, _resume, _rest = log.partition("=== test split fix.v2 --final --resume ===")
    if "TimeoutError: MCP stdio session did not open" in head and "BrokenPipeError" in head:
        key = _crashed_key()
        faults = found.setdefault(key, [])
        for label in ("mcp_timeout", "broken_pipe"):
            if label not in faults:
                faults.append(label)
    return found


def _crashed_key() -> str:
    """The spec after the four results written before the resume."""
    from meridian.bench.splits import load_splits

    resume_at = datetime(2026, 10, 23, 9, 0).astimezone()
    done: set[str] = set()
    directory = _ROOT / "bench/results/test-fix-v2"
    for path in directory.glob("*.json"):
        if path.name in {"summary.json", "LOCK.json"}:
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        if mtime < resume_at:
            done.add(path.stem)
    for spec in load_splits()["test"]:
        if spec.key not in done:
            return spec.key
    raise RuntimeError("no in-flight test key")


def _annotate_infra(record: dict[str, Any], faults: dict[str, list[str]]) -> None:
    names = faults.get(str(record.get("run_key")), [])
    record["infra_faults"] = names
    record["infra_outcome_differs"] = None if names else False
    if names:
        record["infra_note"] = (
            "the failed attempt wrote no result, so an outcome difference was not measured"
        )


def _dev_comparisons() -> list[tuple[str, dict[str, Any]]]:
    found = []
    for path in sorted((_ROOT / "bench/results").glob("dev-fix-*/summary.json")):
        found.append((path.parent.name, json.loads(path.read_text(encoding="utf-8"))))
    return found


def _write_readme() -> None:
    path = _ROOT / "bench/results/test-fix-v2/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Results")
    end = readme.index("## License")
    replacement = render_readme_results(summary).rstrip() + "\n\n"
    (_ROOT / "README.md").write_text(readme[:start] + replacement + readme[end:], encoding="utf-8")


def _load_summary(name: str) -> dict[str, Any] | None:
    path = _ROOT / "bench/results" / name / "summary.json"
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _print_delta(name: str, previous: dict[str, Any], summary: dict[str, Any]) -> None:
    keys = (
        ("detection_recall", ("detection_recall",)),
        ("hard_fp", ("hard_fp",)),
        ("raw_causal", ("root_cause", "raw_causal_correct")),
        ("guarded_causal", ("root_cause", "causal_event_correct")),
        ("kind", ("fix", "kind_appropriate")),
        ("verified", ("fix", "verified")),
        ("false_fix", ("fix", "false_fix")),
        ("escalation", ("fix", "escalation")),
        ("p50", ("ops", "machine_time_p50")),
        ("p95", ("ops", "machine_time_p95")),
        ("usd", ("ops", "usd_per_run")),
    )
    print(f"--- {name} ---")
    for label, path in keys:
        old = _dig(previous, path)
        new = _dig(summary, path)
        print(f"{label}: {_text(old)} -> {_text(new)}")
    old_matrix = _dig(previous, ("critic", "matrix"))
    new_matrix = _dig(summary, ("critic", "matrix"))
    print(f"matrix: {old_matrix} -> {new_matrix}")
    print(
        "match_not_rescored",
        summary["fix"]["match_not_rescored"],
        "infra",
        summary["infra"]["runs_with_faults"]["text"],
    )


def _dig(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _text(value: Any) -> str:
    if isinstance(value, dict) and "text" in value:
        return str(value["text"])
    return str(value)


def _git_head() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    main()
