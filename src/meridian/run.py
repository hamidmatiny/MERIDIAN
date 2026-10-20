"""Run Sentinel → Triage → RootCause → Fix → Critic → Verifier on one partition."""

import argparse
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
from langgraph.checkpoint.sqlite import SqliteSaver

from meridian.agents.critic import CriticChat
from meridian.agents.fake_critic import HeuristicCriticModel
from meridian.agents.fake_fix import HeuristicFixModel
from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.fake_root_cause import HeuristicRootCauseModel
from meridian.agents.live_llm import build_live_chat, build_live_critic
from meridian.agents.root_cause import ToolChat
from meridian.agents.triage import StructuredChat
from meridian.config import MeridianConfigError, Settings, get_settings
from meridian.contracts.fault_type import FaultType
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.partition_ref import PartitionRef
from meridian.contracts.state import MeridianState
from meridian.graph.build import build_graph
from meridian.injector.events import build_events, require_causal_event, write_events
from meridian.injector.faults import INJECTORS
from meridian.mcp_servers.client import StdioSession
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
        chats = _chat(settings, live=args.live)
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
        state = _invoke(settings, chats, ref, root)
    print(state.model_dump_json(indent=2))
    return 0


@dataclass(frozen=True)
class PreparedChats:
    """The four chat models for one run, plus the names recorded on each call."""

    triage_chat: StructuredChat
    triage_model: str
    root_chat: ToolChat
    root_model: str
    fix_chat: ToolChat
    fix_model: str
    critic_chat: CriticChat
    critic_model: str


def _chat(settings: Settings, *, live: bool) -> PreparedChats:
    if not live:
        triage = FakeTriageModel()
        root = HeuristicRootCauseModel()
        fix = HeuristicFixModel()
        critic = HeuristicCriticModel()
        return PreparedChats(
            triage,
            triage.model_name,
            root,
            root.model_name,
            fix,
            fix.model_name,
            critic,
            critic.model_name,
        )
    settings.require_anthropic()
    missing = [
        name
        for name, value in (
            ("MERIDIAN_TRIAGE_MODEL", settings.triage_model),
            ("MERIDIAN_ROOTCAUSE_MODEL", settings.rootcause_model),
            ("MERIDIAN_FIX_MODEL", settings.fix_model),
            ("MERIDIAN_CRITIC_MODEL", settings.critic_model),
        )
        if value is None or value.strip() == ""
    ]
    if missing:
        raise MeridianConfigError("LLM", missing)
    settings.require_xai()
    assert settings.triage_model is not None
    assert settings.rootcause_model is not None
    assert settings.fix_model is not None
    assert settings.critic_model is not None
    return PreparedChats(
        build_live_chat(settings),
        settings.triage_model,
        build_live_chat(
            settings,
            model=settings.rootcause_model,
            model_env="MERIDIAN_ROOTCAUSE_MODEL",
        ),
        settings.rootcause_model,
        build_live_chat(
            settings,
            model=settings.fix_model,
            model_env="MERIDIAN_FIX_MODEL",
        ),
        settings.fix_model,
        build_live_critic(settings),
        settings.critic_model,
    )


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
    label = None
    if intensity == 0:
        frame = clean
        arrived = on_time_arrival(partition)
        events, _causal = build_events(
            dataset=dataset,
            seed=seed,
            fault=None,
            columns=[],
        )
    else:
        frame, label = INJECTORS[fault](
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
        events, _causal = build_events(
            dataset=dataset,
            seed=seed,
            fault=fault,
            columns=list(label.columns_affected),
            intensity=intensity,
        )
    events_path = root / "events.jsonl"
    write_events(events_path, events)
    if label is not None:
        require_causal_event(label, events_path)
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
        generation_seed=seed,
        row_count=rows,
    )


def _invoke(
    settings: Settings,
    chats: PreparedChats,
    ref: PartitionRef,
    evidence_root: Path,
) -> MeridianState:
    checkpoint = Path(settings.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with (
        StdioSession(evidence_root) as session,
        SqliteSaver.from_conn_string(str(checkpoint)) as saver,
    ):
        graph = build_graph(
            num_cpus=settings.ray_num_cpus,
            model_name=chats.triage_model,
            chat=chats.triage_chat,
            checkpointer=saver,
            root_model_name=chats.root_model,
            root_chat=chats.root_chat,
            evidence_root=evidence_root,
            tools=session.tools,
            max_tool_calls=settings.rootcause_max_tool_calls,
            fix_model_name=chats.fix_model,
            fix_chat=chats.fix_chat,
            critic_model_name=chats.critic_model,
            critic_chat=chats.critic_chat,
            sandbox_timeout_s=settings.sandbox_timeout_s,
            sandbox_memory_mb=settings.sandbox_memory_mb,
            max_row_loss=settings.max_row_loss,
            max_quarantine=settings.max_quarantine,
        )
        run_id = uuid4()
        result = graph.invoke(
            MeridianState(run_id=run_id, partitions=[ref]),
            {"configurable": {"thread_id": str(run_id)}},
        )
    return MeridianState.model_validate(result)


if __name__ == "__main__":
    raise SystemExit(main())
