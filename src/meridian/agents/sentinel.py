"""Sentinel. Evaluate partitions on Ray and deduplicate incidents by id. No model calls."""

from datetime import datetime
from pathlib import Path

import ray

from meridian.agents.ray_runtime import ray_runtime
from meridian.contracts.incident import Incident
from meridian.contracts.partition_ref import PartitionRef
from meridian.contracts.reference_profile import ReferenceProfile
from meridian.contracts.state import MeridianState
from meridian.pipeline.parquet import read_parquet
from meridian.quality.evaluate import evaluate


def evaluate_partition(ref: PartitionRef, detected_at: datetime) -> list[Incident]:
    """Load one partition and run the quality contracts."""
    frame = read_parquet(Path(ref.parquet_path))
    profile = ReferenceProfile.model_validate_json(
        Path(ref.reference_profile_path).read_text(encoding="utf-8")
    )
    return evaluate(
        ref.dataset,
        frame,
        ref.metadata,
        profile,
        detected_at=detected_at,
    )


def _remote_entry(ref_json: str, detected_at: str) -> list[str]:
    ref = PartitionRef.model_validate_json(ref_json)
    moment = datetime.fromisoformat(detected_at)
    return [item.model_dump_json() for item in evaluate_partition(ref, moment)]


_REMOTE = ray.remote(_remote_entry)


def dedupe(incidents: list[Incident]) -> list[Incident]:
    """Keep the first incident for each id, then sort by id."""
    chosen: dict[object, Incident] = {}
    for incident in incidents:
        chosen.setdefault(incident.id, incident)
    return [chosen[key] for key in sorted(chosen, key=str)]


def collect_serial(refs: list[PartitionRef], detected_at: datetime) -> list[Incident]:
    """Evaluate partitions in-process. Used to check the Ray result."""
    found: list[Incident] = []
    for ref in refs:
        found.extend(evaluate_partition(ref, detected_at))
    return dedupe(found)


def collect_ray(
    refs: list[PartitionRef],
    detected_at: datetime,
    *,
    num_cpus: int,
) -> list[Incident]:
    """Evaluate each partition as its own Ray task."""
    if not refs:
        return []
    stamp = detected_at.isoformat()
    with ray_runtime(num_cpus):
        pending = [_REMOTE.remote(ref.model_dump_json(), stamp) for ref in refs]
        batches: list[list[str]] = ray.get(pending)
    found: list[Incident] = []
    for batch in batches:
        found.extend(Incident.model_validate_json(raw) for raw in batch)
    return dedupe(found)


def run_sentinel(
    state: MeridianState,
    *,
    num_cpus: int,
    detected_at: datetime,
) -> dict[str, object]:
    """Write deduplicated incidents onto the graph state."""
    incidents = collect_ray(state.partitions, detected_at, num_cpus=num_cpus)
    return {"incidents": incidents}
