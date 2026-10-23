"""Aggregate bench JSON into REPORT.md and summary.json. Rates are n/N."""

import math
from collections import Counter
from typing import Any


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
        "critic": _critic(injected),
        "ops": _ops(runs),
        "forced_final": bool(meta.get("forced_final")),
    }
    if comparisons:
        summary["comparisons"] = [
            {"label": label, "headline": _headline(other)} for label, other in comparisons
        ]
    lines = _markdown(summary, comparisons or [])
    return "\n".join(lines) + "\n", summary


def _headline(summary: dict[str, Any]) -> dict[str, Any]:
    """The columns the README and the side-by-side table show."""
    fix = summary.get("fix", {})
    root = summary.get("root_cause", {})
    ops = summary.get("ops", {})
    return {
        "detection_recall": summary.get("detection_recall"),
        "hard_fp": summary.get("hard_fp"),
        "causal_event_correct": root.get("causal_event_correct"),
        "kind_appropriate": fix.get("kind_appropriate"),
        "verified": fix.get("verified"),
        "false_fix": fix.get("false_fix"),
        "escalation": fix.get("escalation"),
        "machine_time_p50": ops.get("machine_time_p50"),
        "usd_per_run": ops.get("usd_per_run"),
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
    matched = sum(1 for case in verified if case.get("repaired_matches_clean"))
    false_fix = sum(1 for case in verified if not case.get("repaired_matches_clean"))
    escalated = sum(1 for case in rows if case.get("escalated"))
    return {
        "kind_appropriate": _ratio(kind_ok, len(appropriate)),
        "verified": _ratio(len(verified), len(rows)),
        "escalation": _ratio(escalated, len(rows)),
        "repaired_matches_clean_among_verified": _ratio(matched, len(verified)),
        "false_fix": _ratio(false_fix, len(verified)),
    }


def _critic(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [case for case in _cases(runs) if case.get("has_verdict")]
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
    degraded = sum(1 for case in rows if case.get("critic_degraded"))
    corrective = sum(1 for case in rows if case.get("critic_corrective"))
    return {
        "matrix": matrix,
        "degraded": _ratio(degraded, len(rows)),
        "corrective": _ratio(corrective, len(rows)),
        "good_means": "repaired_matches_clean from score_fix",
    }


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
            "",
            "## Fix",
            "",
            f"- kind_appropriate: {_show(summary['fix']['kind_appropriate'])}",
            f"- verified: {_show(summary['fix']['verified'])}",
            f"- escalation: {_show(summary['fix']['escalation'])}",
            "- repaired_matches_clean among verified: "
            f"{_show(summary['fix']['repaired_matches_clean_among_verified'])}",
            f"- FALSE-FIX RATE: {_show(summary['fix']['false_fix'])}",
            "",
            "## Critic",
            "",
            "Good means `repaired_matches_clean` from score_fix.",
            "",
            _matrix(summary["critic"]["matrix"]),
            "",
            f"- degraded: {_show(summary['critic']['degraded'])}",
            f"- corrective: {_show(summary['critic']['corrective'])}",
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
            f"| causal accuracy | {_show(head['causal_event_correct'])} |",
            f"| kind_appropriate | {_show(head['kind_appropriate'])} |",
            f"| verified | {_show(head['verified'])} |",
            f"| false-fix rate | {_show(head['false_fix'])} |",
            f"| escalation rate | {_show(head['escalation'])} |",
            f"| machine time p50 | {head['machine_time_p50']} |",
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
        "causal_event_correct",
        "kind_appropriate",
        "verified",
        "false_fix",
        "escalation",
        "machine_time_p50",
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


def _show(value: object) -> str:
    if isinstance(value, dict) and "text" in value:
        return str(value["text"])
    if value is None:
        return "n/a"
    return str(value)
