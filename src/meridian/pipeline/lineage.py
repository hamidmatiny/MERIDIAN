"""Dataset lineage loaded from ``lineage.yaml``."""

from pathlib import Path

import yaml

from meridian.pipeline.catalog import DATASETS

_EDGES = "edges"
_LINEAGE_PATH = Path(__file__).with_name("lineage.yaml")


def _graph() -> dict[str, tuple[str, ...]]:
    """Children of each known dataset, in file order."""
    raw = yaml.safe_load(_LINEAGE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get(_EDGES), list):
        raise ValueError("lineage.yaml must contain an edges list")
    children: dict[str, list[str]] = {name: [] for name in DATASETS}
    for edge in raw[_EDGES]:
        if not isinstance(edge, dict):
            raise ValueError("each lineage edge must be a mapping")
        upstream = edge.get("upstream")
        downstream = edge.get("downstream")
        if not isinstance(upstream, str) or not isinstance(downstream, str):
            raise ValueError("lineage edges need string upstream and downstream")
        if upstream not in children or downstream not in children:
            raise ValueError(f"lineage names an unknown dataset: {upstream} -> {downstream}")
        children[upstream].append(downstream)
    return {name: tuple(kids) for name, kids in children.items()}


def downstream_of(dataset: str) -> list[str]:
    """Return datasets downstream of ``dataset``, nearest first.

    ``vehicle_telemetry`` reaches both ``trip_summary`` and ``fleet_daily_kpi``.
    """
    graph = _graph()
    if dataset not in graph:
        raise KeyError(f"unknown dataset {dataset!r}")
    ordered: list[str] = []
    seen = {dataset}
    queue = list(graph[dataset])
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        queue.extend(graph[current])
    return ordered


def upstream_of(dataset: str) -> list[str]:
    """Return datasets upstream of ``dataset``, nearest first."""
    graph = _graph()
    if dataset not in graph:
        raise KeyError(f"unknown dataset {dataset!r}")
    parents: dict[str, list[str]] = {name: [] for name in graph}
    for source, children in graph.items():
        for child in children:
            parents[child].append(source)
    ordered: list[str] = []
    seen = {dataset}
    queue = list(parents[dataset])
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        queue.extend(parents[current])
    return ordered
