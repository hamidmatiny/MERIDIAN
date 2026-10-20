"""Build a small partition on disk for graph tests."""

from datetime import datetime
from pathlib import Path

import numpy as np

from meridian.contracts.fault_type import FaultType
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.partition_ref import PartitionRef
from meridian.injector.faults import INJECTORS
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE, generate_dataset
from meridian.pipeline.parquet import write_parquet
from meridian.quality.freshness import late_arrival, on_time_arrival
from meridian.quality.reference import fit_reference_profile


def write_partition(
    root: Path,
    *,
    dataset: str,
    fault: FaultType | None,
    intensity: float,
    seed: int,
    rows: int,
    name: str,
) -> PartitionRef:
    """Write parquet and a holdout profile. ``fault is None`` keeps the frame clean."""
    partition = DEFAULT_PARTITION_DATE
    clean = generate_dataset(dataset, seed, rows=rows, partition_date=partition)
    if fault is None or intensity == 0:
        frame = clean
        arrived = on_time_arrival(partition)
    else:
        frame, _label = INJECTORS[fault](
            clean,
            np.random.default_rng(seed),
            intensity,
            dataset=dataset,
            seed=seed,
        )
        arrived = (
            late_arrival(partition, intensity)
            if fault is FaultType.LATE_PARTITION
            else on_time_arrival(partition)
        )
    holdout_seed = seed + 1
    profile = fit_reference_profile(
        generate_dataset(dataset, holdout_seed, rows=rows, partition_date=partition)
    )
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    parquet_path = folder / f"{dataset}.parquet"
    profile_path = folder / "reference_profile.json"
    write_parquet(frame, parquet_path)
    profile_path.write_text(profile.model_dump_json(), encoding="utf-8")
    return PartitionRef(
        dataset=dataset,
        partition_date=partition,
        parquet_path=str(parquet_path),
        metadata=PartitionMetadata(
            dataset=dataset,
            partition_date=partition,
            arrived_at=arrived,
        ),
        reference_profile_path=str(profile_path),
        generation_seed=seed,
        row_count=rows,
    )


DETECTED_AT = datetime.fromisoformat("2026-10-20T03:00:00+00:00")
