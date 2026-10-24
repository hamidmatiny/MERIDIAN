"""Aggregate bench JSON into REPORT.md and summary.json. Rates are n/N."""

import math
from collections import Counter
from typing import Any

from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident_case import MAX_FIX_ATTEMPTS
from meridian.injector.events import build_events

_OUTCOMES = (
    "VERIFIED_AWAITING_APPROVAL",
    "AGENT_ESCALATED",
    "CRITIC_LIMIT",
    "FIX_ATTEMPT_LIMIT",
    "WALL_BUDGET",
    "NO_PATCH",
    "OTHER",
)


def build_report(
    runs: list[dict[str, Any]],
    *,
    meta: dict[str, Any],
    comparisons: list[tuple[str, dict[str, Any]]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Markdown plus the summary object. Every displayed rate is also in the summary."""
    injected = [run for run in runs if run.get("fault")]
    hard = [run for run in runs if not run.get("fault")]
    summary: dict[str, Any] = {
        "meta": meta,
        "runs": len(runs),
        "detection_recall": _recall(injected),
        "detection_by_fault": _recall_by(injected, "fault"),
        "detection_by_dataset": _recall_by(injected, "dataset"),
        "detection_precision": _precision(injected),
        "hard_fp": _hard_fp(hard),
        "triage": _triage(injected),
        "root_cause": _root(injected),
        "fix": _fix(injected),
        "outcomes": _outcomes(runs),
        "critic": _critic(injected),
        "evidence": _evidence(injected),
        "ops": _ops(runs),
        "infra": _infra(runs),
        "forced_final": bool(meta.get("forced_final")),
        "integrity": _integrity(meta),
    }
    if comparisons:
        summary["comparisons"] = [
            {"label": label, "headline": _headline(other)} for label, other in comparisons
        ]
    lines = _markdown(summary, comparisons or [])
    return "\n".join(lines) + "\n", summary


def _hard_headline(
    integrity: dict[str, Any],
    head: dict[str, Any],
    hard: dict[str, Any] | None,
) -> str:
    if hard is not None:
        field = (hard.get("headline") or {}).get("hard_fp") or hard.get("hard_fp") or {}
        if isinstance(field, dict) and field.get("text"):
            return str(field["text"])
    value = integrity.get("hard_mode_fp") or head["hard_fp"]
    return str(value)


def render_limitations(summary: dict[str, Any]) -> str:
    """Limitation lines whose counts come from the summary."""
    categories = summary["outcomes"]["categories"]
    evidence = summary["evidence"]
    runs = int(summary["runs"])
    one = evidence["exactly_one"]
    single = one["text"] if isinstance(one, dict) else f"{one}/{runs}"
    return "\n".join(
        [
            "- synthetic benchmark",
            f"- easy evidence world ({single} single candidate)",
            (
                "- FIX_ATTEMPT_LIMIT is the dominant failure "
                f"({categories['FIX_ATTEMPT_LIMIT']}/{runs})"
            ),
            "- local models only",
            "- sandbox limits on macOS",
        ]
    )


def _headline(summary: dict[str, Any]) -> dict[str, Any]:
    """The columns the README and the side-by-side table show."""
    fix = summary.get("fix", {})
    root = summary.get("root_cause", {})
    ops = summary.get("ops", {})
    return {
        "detection_recall": summary.get("detection_recall"),
        "hard_fp": summary.get("hard_fp"),
        "raw_causal_correct": root.get("raw_causal_correct"),
        "causal_event_correct": root.get("causal_event_correct"),
        "kind_appropriate": fix.get("kind_appropriate"),
        "verified": fix.get("verified"),
        "false_fix": fix.get("false_fix"),
        "escalation": fix.get("escalation"),
        "machine_time_p50": ops.get("machine_time_p50"),
        "machine_time_p95": ops.get("machine_time_p95"),
        "usd_per_run": ops.get("usd_per_run"),
        "critic_matrix": summary.get("critic", {}).get("matrix"),
    }


def _recall(runs: list[dict[str, Any]]) -> dict[str, Any]:
    hit = 0
    for run in runs:
        faults = {item["fault"] for item in run.get("incidents", [])}
        if run.get("fault") in faults:
            hit += 1
    return _ratio(hit, len(runs))


def _recall_by(runs: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run.get(field)), []).append(run)
    return {name: _recall(items) for name, items in sorted(grouped.items())}


def _precision(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Incidents whose fault equals the injected fault, per fault and per dataset."""
    per_fault: dict[str, list[bool]] = {}
    per_dataset: dict[str, list[bool]] = {}
    for run in runs:
        injected = run.get("fault")
        for incident in run.get("incidents", []):
            match = incident.get("fault") == injected
            per_fault.setdefault(str(incident.get("fault")), []).append(match)
            per_dataset.setdefault(str(run.get("dataset")), []).append(match)
    return {
        "by_fault": {
            name: _ratio(sum(flags), len(flags)) for name, flags in sorted(per_fault.items())
        },
        "by_dataset": {
            name: _ratio(sum(flags), len(flags)) for name, flags in sorted(per_dataset.items())
        },
    }


def _hard_fp(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fired = sum(1 for run in runs if run.get("incidents"))
    incidents = sum(len(run.get("incidents", [])) for run in runs)
    return {**_ratio(fired, len(runs)), "incidents": incidents}


def _triage(runs: list[dict[str, Any]]) -> dict[str, Any]:
    severities: Counter[str] = Counter()
    raised = 0
    seen = 0
    for case in _cases(runs):
        if case.get("severity") is None:
            continue
        seen += 1
        severities[str(case["severity"])] += 1
        if case.get("floor_raised"):
            raised += 1
    return {
        "distribution": dict(severities),
        "floor_raise": _ratio(raised, seen),
        "note": "No ground-truth severity exists. This is not an accuracy.",
    }


def _root(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [case for case in _cases(runs) if case.get("has_hypothesis")]
    raw = sum(1 for case in rows if case.get("raw_causal_correct"))
    causal = sum(1 for case in rows if case.get("causal_event_correct"))
    guard = sum(1 for case in rows if case.get("guardrail_fn"))
    jaccard = [
        float(case["columns_jaccard"]) for case in rows if case.get("columns_jaccard") is not None
    ]
    violations = [int(case.get("grounding_violations") or 0) for case in rows]
    mean_j = None if not jaccard else sum(jaccard) / len(jaccard)
    mean_v = None if not violations else sum(violations) / len(violations)
    return {
        "raw_causal_correct": _ratio(raw, len(rows)),
        "causal_event_correct": _ratio(causal, len(rows)),
        "guardrail_false_negative": _ratio(guard, len(rows)),
        "mean_columns_jaccard": mean_j,
        "mean_grounding_violations": mean_v,
    }


def _fix(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _cases(runs)
    appropriate = [case for case in rows if case.get("kind_appropriate") is not None]
    kind_ok = sum(1 for case in appropriate if case.get("kind_appropriate"))
    verified = [case for case in rows if case.get("verified")]
    scored = [case for case in verified if _match_known(case)]
    matched = sum(1 for case in scored if case.get("repaired_matches_clean"))
    false_fix = sum(1 for case in scored if case.get("repaired_matches_clean") is False)
    escalated = sum(1 for case in rows if case.get("escalated"))
    return {
        "kind_appropriate": _ratio(kind_ok, len(appropriate)),
        "verified": _ratio(len(verified), len(rows)),
        "escalation": _ratio(escalated, len(rows)),
        "repaired_matches_clean_among_verified": _ratio(matched, len(scored)),
        "false_fix": _ratio(false_fix, len(scored)),
        "match_not_rescored": sum(1 for case in rows if not _match_known(case)),
    }


def _critic(runs: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = [case for case in _cases(runs) if case.get("has_verdict")]
    rows = [case for case in verdicts if _match_known(case)]
    matrix = {
        "approved_good": 0,
        "approved_bad": 0,
        "rejected_good": 0,
        "rejected_bad": 0,
    }
    for case in rows:
        good = bool(case.get("repaired_matches_clean"))
        if case.get("critic_approved"):
            matrix["approved_good" if good else "approved_bad"] += 1
        else:
            matrix["rejected_good" if good else "rejected_bad"] += 1
    degraded = sum(1 for case in verdicts if case.get("critic_degraded"))
    corrective = sum(1 for case in verdicts if case.get("critic_corrective"))
    bad = matrix["approved_bad"] + matrix["rejected_bad"]
    approved = matrix["approved_good"] + matrix["approved_bad"]
    blocked = sum(
        1
        for case in _cases(runs)
        if case.get("critic_approved") is True
        and case.get("repaired_matches_clean") is False
        and not case.get("verified")
    )
    return {
        "matrix": matrix,
        "degraded": _ratio(degraded, len(verdicts)),
        "corrective": _ratio(corrective, len(verdicts)),
        "unscored": len(verdicts) - len(rows),
        "bad_patch_catch_rate": _ratio(matrix["rejected_bad"], bad),
        "approval_precision": _ratio(matrix["approved_good"], approved),
        "verifier_blocked_bad": blocked,
        "good_means": "repaired_matches_clean from score_fix",
    }


def _outcomes(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """One category per run. The counts sum to the number of runs."""
    counts: Counter[str] = Counter()
    reasons: list[dict[str, str]] = []
    for run in runs:
        name, reason = _outcome(run)
        counts[name] += 1
        if name == "OTHER":
            reasons.append({"run_key": str(run.get("run_key")), "reason": reason})
    categories = {name: counts[name] for name in _OUTCOMES}
    total = sum(categories.values())
    if total != len(runs):
        raise RuntimeError(f"outcome categories sum to {total}, not {len(runs)}")
    return {"N": len(runs), "categories": categories, "other": reasons}


def _outcome(run: dict[str, Any]) -> tuple[str, str]:
    if run.get("wall_abort") or "WALL_BUDGET" in str(run.get("abort_reason") or ""):
        return "WALL_BUDGET", ""
    cases = list(run.get("cases") or [])
    if any(case.get("verified") for case in cases):
        return "VERIFIED_AWAITING_APPROVAL", ""
    if any(case.get("escalated") for case in cases):
        return "AGENT_ESCALATED", ""
    if cases and all(case.get("kind") is None for case in cases):
        return "NO_PATCH", ""
    if any(case.get("has_verdict") and case.get("critic_approved") is False for case in cases):
        return "CRITIC_LIMIT", ""
    fix_calls = int(((run.get("structured") or {}).get("fix") or {}).get("n") or 0)
    if fix_calls >= MAX_FIX_ATTEMPTS:
        return "FIX_ATTEMPT_LIMIT", ""
    if not cases:
        return "OTHER", str(run.get("abort_reason") or "no incident")
    return "OTHER", str(run.get("abort_reason") or "stopped without a limit signal")


def _evidence(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """How many log events share the causal event's kind and dataset."""
    counts: Counter[int] = Counter()
    for run in runs:
        if not run.get("fault") or run.get("seed") is None or run.get("intensity") is None:
            continue
        counts[_candidate_count(run)] += 1
    total = sum(counts.values())
    exactly_one = counts.get(1, 0)
    easy = total > 0 and exactly_one * 2 > total
    note = "causal accuracy reflects an easy evidence world" if easy else ""
    return {
        "distribution": {str(count): counts[count] for count in sorted(counts)},
        "exactly_one": _ratio(exactly_one, total),
        "easy_evidence": easy,
        "note": note,
    }


def _candidate_count(run: dict[str, Any]) -> int:
    events, causal_id = build_events(
        dataset=str(run["dataset"]),
        seed=int(run["seed"]),
        fault=FaultType(str(run["fault"])),
        columns=["column"],
        intensity=float(run["intensity"]),
    )
    causal = next(event for event in events if event.id == causal_id)
    return sum(
        1 for event in events if event.kind == causal.kind and event.dataset == causal.dataset
    )


def _ops(runs: list[dict[str, Any]]) -> dict[str, Any]:
    machine = [
        float(run["machine_time_s"]) for run in runs if run.get("machine_time_s") is not None
    ]
    walls = [float(run.get("wall_s") or 0) for run in runs]
    tokens = sum(
        int(run.get("input_tokens") or 0) + int(run.get("output_tokens") or 0) for run in runs
    )
    usd = sum(float(run.get("usd") or 0) for run in runs)
    timeouts = sum(int(run.get("timeouts") or 0) for run in runs)
    aborts = sum(1 for run in runs if run.get("wall_abort"))
    latency: dict[str, list[float]] = {}
    structured: dict[str, list[int]] = {}
    for run in runs:
        for node, values in (run.get("node_latency_ms") or {}).items():
            latency.setdefault(node, []).extend(float(value) for value in values)
        for node, slot in (run.get("structured") or {}).items():
            bucket = structured.setdefault(node, [0, 0])
            bucket[0] += int(slot.get("ok", 0))
            bucket[1] += int(slot.get("n", 0))
    per_node = {
        node: {"p50": _percentile(values, 50), "p95": _percentile(values, 95)}
        for node, values in sorted(latency.items())
    }
    structured_rates = {node: _ratio(ok, count) for node, (ok, count) in sorted(structured.items())}
    count = len(runs)
    return {
        "machine_time_p50": _percentile(machine, 50),
        "machine_time_p95": _percentile(machine, 95),
        "machine_time_n": len(machine),
        "wall_p50": _percentile(walls, 50),
        "wall_p95": _percentile(walls, 95),
        "tokens_per_run": None if count == 0 else tokens / count,
        "usd_total": usd,
        "usd_per_run": None if count == 0 else usd / count,
        "node_latency_ms": per_node,
        "timeouts": timeouts,
        "wall_aborts": _ratio(aborts, count),
        "structured_success_rate": structured_rates,
    }


def _match_known(case: dict[str, Any]) -> bool:
    """A missing match is not a failure. It was not re-scored."""
    return "repaired_matches_clean" in case and case["repaired_matches_clean"] is not None


def _infra(runs: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [run for run in runs if run.get("infra_faults")]
    differ = [run for run in flagged if run.get("infra_outcome_differs") is True]
    return {
        "runs_with_faults": _ratio(len(flagged), len(runs)),
        "outcomes_differ": _ratio(len(differ), len(flagged)),
        "outcomes_unmeasured": sum(
            1 for run in flagged if run.get("infra_outcome_differs") is None
        ),
        "runs": [
            {
                "run_key": run.get("run_key"),
                "faults": list(run.get("infra_faults") or []),
                "outcome_differs": run.get("infra_outcome_differs"),
            }
            for run in flagged
        ],
    }


def _integrity(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "scorer_sha256": meta.get("scorer_sha256"),
        "scorer_parent_commit": meta.get("scorer_parent_commit"),
        "mixed_version": bool(meta.get("mixed_version")),
        "started_dirty": bool(meta.get("started_dirty")),
        "hard_mode_fp": meta.get("hard_mode_fp"),
        "label": meta.get("label") or "",
    }


def _cases(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for run in runs:
        for case in run.get("cases", []):
            found.append(case)
    return found


def _ratio(n: int, total: int) -> dict[str, Any]:
    return {"n": n, "N": total, "text": f"{n}/{total}"}


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percent / 100 * len(ordered)) - 1))
    return ordered[index]


def _markdown(summary: dict[str, Any], comparisons: list[tuple[str, dict[str, Any]]]) -> list[str]:
    meta = summary["meta"]
    lines = [
        f"# MERIDIAN-Bench {meta.get('split')}",
        "",
        f"- commit: `{meta.get('commit')}`",
        f"- models: `{meta.get('models')}`",
        f"- prompt versions: `{meta.get('prompt_versions')}`",
        f"- ollama: `{meta.get('ollama')}`",
        f"- seeds and split: `{meta.get('seeds')}` / {meta.get('split')}",
        f"- forced second final: {summary['forced_final']}",
        *_integrity_lines(summary),
        "",
        "## Headline",
        "",
        _headline_table(summary),
        "",
    ]
    if comparisons:
        lines.extend(["## Dev and test", "", _comparison_table(comparisons, summary), ""])
    lines.extend(
        [
            "## Detection",
            "",
            f"- recall: {_show(summary['detection_recall'])}",
            f"- hard-mode false-positive runs: {_show(summary['hard_fp'])}",
            f"- hard-mode incident count: {summary['hard_fp'].get('incidents', 0)}",
            "",
            "### Recall by fault",
            "",
            _mapping(summary["detection_by_fault"]),
            "",
            "### Recall by dataset",
            "",
            _mapping(summary["detection_by_dataset"]),
            "",
            "### Precision by fault",
            "",
            _mapping(summary["detection_precision"]["by_fault"]),
            "",
            "## Triage",
            "",
            summary["triage"]["note"],
            f"- floor-raise: {_show(summary['triage']['floor_raise'])}",
            f"- severity distribution: `{summary['triage']['distribution']}`",
            "",
            "## Root cause",
            "",
            f"- raw_causal_correct: {_show(summary['root_cause']['raw_causal_correct'])}",
            f"- causal_event_correct: {_show(summary['root_cause']['causal_event_correct'])}",
            "- guardrail false-negative: "
            f"{_show(summary['root_cause']['guardrail_false_negative'])}",
            f"- mean columns_jaccard: {summary['root_cause']['mean_columns_jaccard']}",
            f"- mean grounding violations: {summary['root_cause']['mean_grounding_violations']}",
            _evidence_line(summary["evidence"]),
            "",
            "## Fix",
            "",
            f"- kind_appropriate: {_show(summary['fix']['kind_appropriate'])}",
            f"- verified: {_show(summary['fix']['verified'])}",
            f"- escalation: {_show(summary['fix']['escalation'])}",
            "- repaired_matches_clean among verified: "
            f"{_show(summary['fix']['repaired_matches_clean_among_verified'])}",
            f"- FALSE-FIX RATE: {_show(summary['fix']['false_fix'])}",
            f"- match not re-scored: {summary['fix']['match_not_rescored']}",
            "",
            "## Outcomes",
            "",
            _outcome_lines(summary["outcomes"]),
            "",
            "## Infra",
            "",
            f"- runs with infra faults: {_show(summary['infra']['runs_with_faults'])}",
            f"- of those, outcomes differ: {_show(summary['infra']['outcomes_differ'])}",
            f"- outcomes unmeasured: {summary['infra']['outcomes_unmeasured']}",
            _infra_lines(summary["infra"]["runs"]),
            "",
            "## Critic",
            "",
            "Good means `repaired_matches_clean` from score_fix.",
            "",
            _matrix(summary["critic"]["matrix"]),
            "",
            f"- degraded: {_show(summary['critic']['degraded'])}",
            f"- corrective: {_show(summary['critic']['corrective'])}",
            f"- verdicts with no re-scored match: {summary['critic']['unscored']}",
            f"- bad-patch catch rate: {_show(summary['critic']['bad_patch_catch_rate'])}",
            f"- approval precision: {_show(summary['critic']['approval_precision'])}",
            "- bad approved patches blocked by the verifier: "
            f"{summary['critic']['verifier_blocked_bad']}",
            "",
            "## Ops",
            "",
            f"- machine_time_s p50: {summary['ops']['machine_time_p50']}",
            f"- machine_time_s p95: {summary['ops']['machine_time_p95']}",
            f"- machine_time_s n: {summary['ops']['machine_time_n']}",
            f"- tokens per run: {summary['ops']['tokens_per_run']}",
            f"- USD per run: {summary['ops']['usd_per_run']}",
            f"- timeouts: {summary['ops']['timeouts']}",
            f"- WALL_BUDGET aborts: {_show(summary['ops']['wall_aborts'])}",
            f"- per-node latency ms: `{summary['ops']['node_latency_ms']}`",
            f"- structured_success_rate: `{summary['ops']['structured_success_rate']}`",
            "",
        ]
    )
    return lines


def _headline_table(summary: dict[str, Any]) -> str:
    head = _headline(summary)
    return "\n".join(
        [
            "| metric | value |",
            "| --- | --- |",
            f"| detection recall | {_show(head['detection_recall'])} |",
            f"| hard-mode FP rate | {_show(head['hard_fp'])} |",
            f"| raw causal accuracy | {_show(head['raw_causal_correct'])} |",
            f"| guarded causal accuracy | {_show(head['causal_event_correct'])} |",
            f"| kind_appropriate | {_show(head['kind_appropriate'])} |",
            f"| verified | {_show(head['verified'])} |",
            f"| false-fix rate | {_show(head['false_fix'])} |",
            f"| escalation rate | {_show(head['escalation'])} |",
            f"| critic confusion matrix | {_matrix_cell(head.get('critic_matrix'))} |",
            f"| machine time p50 | {head['machine_time_p50']} |",
            f"| machine time p95 | {head['machine_time_p95']} |",
            f"| USD per run | {head['usd_per_run']} |",
        ]
    )


def _comparison_table(
    comparisons: list[tuple[str, dict[str, Any]]],
    current: dict[str, Any],
) -> str:
    rows = [(label, _headline(item)) for label, item in comparisons]
    rows.append((f"{current['meta'].get('split')} current", _headline(current)))
    header = "| metric | " + " | ".join(label for label, _head in rows) + " |"
    rule = "| --- | " + " | ".join("---" for _row in rows) + " |"
    keys = [
        "detection_recall",
        "raw_causal_correct",
        "causal_event_correct",
        "kind_appropriate",
        "verified",
        "false_fix",
        "escalation",
        "machine_time_p50",
        "machine_time_p95",
        "usd_per_run",
    ]
    body = []
    for key in keys:
        cells = " | ".join(_show(head.get(key)) for _label, head in rows)
        body.append(f"| {key} | {cells} |")
    return "\n".join([header, rule, *body])


def _mapping(values: dict[str, dict[str, Any]]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {name}: {_show(ratio)}" for name, ratio in values.items())


def _matrix(matrix: dict[str, int]) -> str:
    return "\n".join(
        [
            "| | good | bad |",
            "| --- | --- | --- |",
            f"| approved | {matrix['approved_good']} | {matrix['approved_bad']} |",
            f"| rejected | {matrix['rejected_good']} | {matrix['rejected_bad']} |",
        ]
    )


def _outcome_lines(outcomes: dict[str, Any]) -> str:
    categories = outcomes["categories"]
    lines = [f"- {name}: {categories[name]}" for name in _OUTCOMES]
    lines.append(f"- sum: {outcomes['N']}")
    for item in outcomes["other"]:
        lines.append(f"- OTHER `{item['run_key']}`: {item['reason']}")
    return "\n".join(lines)


def _evidence_line(evidence: dict[str, Any]) -> str:
    lines = [
        f"- evidence candidates by kind and dataset: `{evidence['distribution']}`",
        f"- runs with exactly one candidate: {_show(evidence['exactly_one'])}",
    ]
    if evidence.get("note"):
        lines.append(f"- {evidence['note']}")
    return "\n".join(lines)


def _integrity_lines(summary: dict[str, Any]) -> list[str]:
    info = summary.get("integrity") or {}
    lines: list[str] = []
    label = info.get("label")
    if label:
        lines.append(f"- {label}")
    digest = info.get("scorer_sha256")
    parent = info.get("scorer_parent_commit")
    if digest:
        lines.append(
            "- every score was computed by scorer version "
            f"`{digest}` of src/meridian/bench/score_fix.py "
            f"(parent commit `{parent}`)"
        )
    if info.get("mixed_version"):
        lines.append("- mixed-version test run")
    if info.get("started_dirty"):
        lines.append("- this test run started from a dirty tree. LOCK.json was not rewritten.")
    return lines


def _infra_lines(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "- none"
    return "\n".join(
        f"- `{item['run_key']}`: {', '.join(item['faults'])}; "
        f"outcome_differs={item['outcome_differs']}"
        for item in runs
    )


def _matrix_cell(matrix: object) -> str:
    if not isinstance(matrix, dict):
        return "n/a"
    return (
        f"approved {matrix.get('approved_good', 0)}/{matrix.get('approved_bad', 0)}; "
        f"rejected {matrix.get('rejected_good', 0)}/{matrix.get('rejected_bad', 0)}"
    )


def render_readme_results(
    summary: dict[str, Any],
    *,
    hard: dict[str, Any] | None = None,
) -> str:
    """README Results copied from one summary. No number is introduced here."""
    head = _headline(summary)
    integrity = summary.get("integrity") or {}
    critic = summary.get("critic") or {}
    evidence = summary.get("evidence") or {}
    lines = [
        "## Results",
        "",
        "This is a synthetic benchmark.",
        "",
        "Numbers below are copied from `bench/results/test-fix-v2-clean/summary.json`. "
        "The full write-up is "
        "[bench/results/test-fix-v2-clean/REPORT.md](bench/results/test-fix-v2-clean/REPORT.md).",
        "",
        "The earlier run in "
        "[bench/results/test-fix-v2/REPORT.md](bench/results/test-fix-v2/REPORT.md) "
        "is a dirty-tree run, superseded.",
        "",
    ]
    if integrity.get("scorer_sha256"):
        lines.append(
            "Every score was computed by scorer version "
            f"`{integrity.get('scorer_sha256')}` of src/meridian/bench/score_fix.py "
            f"(parent commit `{integrity.get('scorer_parent_commit')}`)."
        )
        lines.append("")
    if integrity.get("mixed_version"):
        lines.append("mixed-version test run")
        lines.append("")
    if integrity.get("started_dirty"):
        lines.append("started from a dirty tree")
        lines.append("")
    hard_fp = _hard_headline(integrity, head, hard)
    lines.extend(
        [
            "| metric | value |",
            "| --- | --- |",
            f"| detection recall | {_show(head['detection_recall'])} |",
            f"| hard-mode FP rate | {_show(hard_fp)} |",
            f"| raw causal accuracy | {_show(head['raw_causal_correct'])} |",
            f"| guarded causal accuracy | {_show(head['causal_event_correct'])} |",
            f"| kind_appropriate | {_show(head['kind_appropriate'])} |",
            f"| verified | {_show(head['verified'])} |",
            f"| false-fix rate | {_show(head['false_fix'])} |",
            f"| escalation rate | {_show(head['escalation'])} |",
            f"| critic confusion matrix | {_matrix_cell(head.get('critic_matrix'))} |",
            f"| machine time p50 | {head['machine_time_p50']} |",
            f"| machine time p95 | {head['machine_time_p95']} |",
            f"| USD per run | {head['usd_per_run']} |",
            "",
        ]
    )
    if hard is not None:
        commit = str((hard.get("meta") or {}).get("commit") or "")
        lines.append(
            "Hard-mode FP is copied from `bench/results/hard-clean/summary.json` "
            f"at commit `{commit}`."
        )
        lines.append("")
    outcomes = summary.get("outcomes")
    if outcomes:
        lines.extend(["## Outcome breakdown", "", _outcome_lines(outcomes), ""])
    if critic.get("bad_patch_catch_rate") is not None:
        blocked = critic.get("verifier_blocked_bad")
        lines.extend(
            [
                f"- bad-patch catch rate: {_show(critic.get('bad_patch_catch_rate'))}",
                f"- approval precision: {_show(critic.get('approval_precision'))}",
                "Safety comes from the deterministic verifier: "
                f"{blocked} bad patches approved by the LLM critic were "
                "blocked before reaching a human.",
                "",
            ]
        )
    if evidence.get("note"):
        lines.append(str(evidence["note"]))
        lines.append(f"- evidence distribution: `{evidence.get('distribution')}`")
        lines.append(f"- exactly one candidate: {_show(evidence.get('exactly_one'))}")
        lines.append("")
    if summary.get("comparisons"):
        lines.extend(
            [
                "Dev versus test, from the comparisons stored on the test summary:",
                "",
                _readme_comparison(summary),
                "",
            ]
        )
    return "\n".join(lines)


def _readme_comparison(summary: dict[str, Any]) -> str:
    stored = summary.get("comparisons") or []
    rows = [(str(item["label"]), item["headline"]) for item in stored]
    rows.append(("test-fix-v2-clean", _headline(summary)))
    header = "| metric | " + " | ".join(label for label, _head in rows) + " |"
    rule = "| --- | " + " | ".join("---" for _row in rows) + " |"
    keys = [
        "detection_recall",
        "raw_causal_correct",
        "causal_event_correct",
        "kind_appropriate",
        "verified",
        "false_fix",
        "escalation",
        "machine_time_p50",
        "machine_time_p95",
        "usd_per_run",
    ]
    body = [
        f"| {key} | " + " | ".join(_show(head.get(key)) for _label, head in rows) + " |"
        for key in keys
    ]
    return "\n".join([header, rule, *body])


def _show(value: object) -> str:
    if isinstance(value, dict) and "text" in value:
        return str(value["text"])
    if value is None:
        return "n/a"
    return str(value)
