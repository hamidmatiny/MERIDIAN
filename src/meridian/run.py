"""Run Sentinel → Triage on one injected partition."""

import argparse
import sys
import tempfile
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
from langgraph.checkpoint.sqlite import SqliteSaver

from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.live_llm import build_live_chat
from meridian.agents.triage import StructuredChat
from meridian.config import MeridianConfigError, Settings, get_settings
from meridian.contracts.fault_type import FaultType
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.partition_ref import PartitionRef
from meridian.contracts.state import MeridianState
from meridian.graph.build import build_graph
from meridian.injector.faults import INJECTORS
from meridian.pipeline.catalog import DATASETS, DEFAULT_ROWS
from meridian.pipeline.generate import DEFAULT_PARTITION_DATE, generate_dataset
from meridian.pipeline.parquet import write_parquet
from meridian.quality.freshness import late_arrival, on_time_arrival
from meridian.quality.reference import fit_reference_profile


def main(argv: list[str] | None = None) -> int:
    """Inject one partition, run the graph, and print the final state as JSON."""
    parser = argparse.ArgumentParser(prog="meridian.run")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--fault", required=True, choices=[item.value for item in FaultType])
    parser.add_argument("--intensity", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fake-llm", action="store_true", help="Deterministic fake model. Default.")
    mode.add_argument("--live", action="store_true", help="Call Claude. Requires API keys.")
    args = parser.parse_args(argv)
    if args.intensity < 0 or args.intensity > 1:
        parser.error("intensity must be in the interval [0, 1]")
    if args.seed < 0:
        parser.error("seed must be >= 0")
    settings = get_settings()
    try:
        chat, model_name = _chat(settings, live=args.live)
    except MeridianConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    fault = FaultType(args.fault)
    partition = DEFAULT_PARTITION_DATE
    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        ref = _prepare(
            root,
            dataset=args.dataset,
            fault=fault,
            intensity=args.intensity,
            seed=args.seed,
            rows=args.rows,
            partition=partition,
        )
        state = _invoke(settings, chat, model_name, ref)
    print(state.model_dump_json(indent=2))
    return 0


def _chat(settings: Settings, *, live: bool) -> tuple[StructuredChat, str]:
    if not live:
        fake = FakeTriageModel()
        return fake, fake.model_name
    settings.require_anthropic()
    live_chat = build_live_chat(settings)
    assert settings.triage_model is not None
    return live_chat, settings.triage_model


def _prepare(
    root: Path,
    *,
    dataset: str,
    fault: FaultType,
    intensity: float,
    seed: int,
    rows: int,
    partition: date,
) -> PartitionRef:
    clean = generate_dataset(dataset, seed, rows=rows, partition_date=partition)
    if intensity == 0:
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
    holdout = generate_dataset(dataset, seed + 1, rows=rows, partition_date=partition)
    profile = fit_reference_profile(holdout)
    parquet_path = root / f"{dataset}.parquet"
    profile_path = root / "reference_profile.json"
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
    )


def _invoke(
    settings: Settings,
    chat: StructuredChat,
    model_name: str,
    ref: PartitionRef,
) -> MeridianState:
    checkpoint = Path(settings.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
        graph = build_graph(
            num_cpus=settings.ray_num_cpus,
            model_name=model_name,
            chat=chat,
            checkpointer=saver,
        )
        run_id = uuid4()
        result = graph.invoke(
            MeridianState(run_id=run_id, partitions=[ref]),
            {"configurable": {"thread_id": str(run_id)}},
        )
    return MeridianState.model_validate(result)


if __name__ == "__main__":
    raise SystemExit(main())
