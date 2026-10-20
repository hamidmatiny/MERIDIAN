"""Command line for a single seeded injection."""

import argparse
from pathlib import Path

import numpy as np

from meridian.contracts.fault_type import FaultType
from meridian.injector.faults import INJECTORS
from meridian.pipeline.catalog import DATASETS, DEFAULT_ROWS
from meridian.pipeline.generate import generate_dataset
from meridian.pipeline.parquet import write_parquet


def main(argv: list[str] | None = None) -> int:
    """Write a corrupted Parquet file and its ``injected_fault.json`` label."""
    parser = argparse.ArgumentParser(prog="meridian.injector")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--fault",
        required=True,
        choices=[item.value for item in FaultType],
    )
    parser.add_argument("--intensity", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    args = parser.parse_args(argv)
    fault = FaultType(args.fault)
    frame = generate_dataset(args.dataset, args.seed, args.rows)
    rng = np.random.default_rng(args.seed)
    corrupted, label = INJECTORS[fault](
        frame,
        rng,
        args.intensity,
        dataset=args.dataset,
        seed=args.seed,
    )
    destination = args.out
    destination.mkdir(parents=True, exist_ok=True)
    parquet_path = destination / f"{args.dataset}.parquet"
    label_path = destination / "injected_fault.json"
    write_parquet(corrupted, parquet_path)
    label_path.write_text(label.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(parquet_path)
    print(label_path)
    return 0
