"""Run Sentinel → Triage → RootCause → Fix → Critic → Verifier on one partition."""

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver

from meridian.agents.critic import CriticChat
from meridian.agents.fake_critic import HeuristicCriticModel
from meridian.agents.fake_fix import HeuristicFixModel
from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.fake_root_cause import HeuristicRootCauseModel
from meridian.agents.ray_runtime import shared_ray
from meridian.agents.root_cause import ToolChat
from meridian.agents.triage import StructuredChat
from meridian.bench.score_fix import score as score_fix
from meridian.bench.score_root_cause import score as score_root_cause
from meridian.config import MeridianConfigError, Settings, get_settings
from meridian.contracts.fault_type import FaultType
from meridian.contracts.injected_fault import InjectedFault
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.partition_ref import PartitionRef
from meridian.contracts.state import MeridianState
from meridian.graph.build import build_graph
from meridian.injector.events import build_events, require_causal_event, write_events
from meridian.injector.faults import INJECTORS
from meridian.llm.router import build_prepared
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
    mode.add_argument(
        "--live",
        action="store_true",
        help="Call the configured models. Defaults to the free tier.",
    )
    args = parser.parse_args(argv)
    if args.intensity < 0 or args.intensity > 1:
        parser.error("intensity must be in the interval [0, 1]")
    if args.seed < 0:
        parser.error("seed must be >= 0")
    try:
        settings = get_settings()
        chats = _chat(settings, live=args.live)
    except MeridianConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    fault = FaultType(args.fault)
    partition = DEFAULT_PARTITION_DATE
    with tempfile.TemporaryDirectory() as tmp_name:
        root = Path(tmp_name)
        prepared = _prepare(
            root,
            dataset=args.dataset,
            fault=fault,
            intensity=args.intensity,
            seed=args.seed,
            rows=args.rows,
            partition=partition,
        )
        started = time.perf_counter()
        state = _invoke(settings, chats, prepared.ref, root)
        wall_s = time.perf_counter() - started
    if args.live:
        print(
            _report(state, wall_s=wall_s, clean=prepared.clean, label=prepared.label),
            file=sys.stderr,
        )
    print(state.model_dump_json(indent=2))
    if state.aborted:
        print(f"aborted: {state.abort_reason}", file=sys.stderr)
        return 1
    return 0


@dataclass(frozen=True)
class PreparedPartition:
    """The partition the graph reads, plus the bench-only clean frame and label."""

    ref: PartitionRef
    clean: object
    label: InjectedFault | None


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
    built = build_prepared(settings)
    return PreparedChats(
        built[0][0],
        built[0][1],
        built[1][0],
        built[1][1],
        built[2][0],
        built[2][1],
        built[3][0],
        built[3][1],
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
) -> PreparedPartition:
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
    return PreparedPartition(
        ref=PartitionRef(
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
        ),
        clean=clean,
        label=label,
    )


def _report(
    state: MeridianState,
    *,
    wall_s: float,
    clean: object,
    label: InjectedFault | None,
) -> str:
    """Human-readable summary of one live run. Numbers come from the state, not a guess."""
    lines = [f"aborted={state.aborted} reason={state.abort_reason or '-'}"]
    by_node: dict[str, list[str]] = {}
    for call in state.llm_calls:
        usd = "0.00" if call.provider not in {"anthropic", "xai"} else "paid"
        by_node.setdefault(call.node, []).append(
            f"provider={call.provider} model={call.model} fallback_used={call.fallback_used} "
            f"input_tokens={call.input_tokens} output_tokens={call.output_tokens} "
            f"prompt_tokens_before={call.prompt_tokens_before} "
            f"prompt_tokens_after={call.prompt_tokens_after} usd={usd}"
        )
    for node, rows in by_node.items():
        lines.append(f"node {node}")
        lines.extend(f"  {row}" for row in rows)
    for case in state.cases.values():
        triage = case.triage
        severity = None
        if triage is not None and triage.llm_severity is not None:
            severity = triage.llm_severity.value
        succeeded = triage is not None and not triage.triage_degraded
        lines.append(f"triage succeeded={succeeded} llm_severity={severity}")
        hypothesis = case.hypothesis
        if hypothesis is not None:
            lines.append(
                f"hypothesis={hypothesis.id} mechanism={hypothesis.mechanism} "
                f"grounding_violations={hypothesis.grounding_violations}"
            )
        if case.dry_run is not None:
            lines.append(
                "dry_run "
                f"rows_changed={case.dry_run.rows_changed} "
                f"rows_changed_outside={case.dry_run.rows_changed_outside} "
                f"over_broad={case.dry_run.over_broad} "
                f"contracts_after={case.dry_run.contracts_after} "
                f"psi_after={case.dry_run.psi_after}"
            )
        for verdict in case.verdicts:
            lines.append(
                f"verdict round={verdict.round} approved={verdict.approved} "
                f"critic_degraded={verdict.critic_degraded} "
                f"objections={[item.kind.value for item in verdict.objections]}"
            )
        if case.verification is not None:
            checks = ",".join(
                f"{item.name}:{'pass' if item.passed else 'fail'}"
                for item in case.verification.checks
            )
            lines.append(f"verifier passed={case.verification.passed} checks={checks}")
        lines.append(f"needs_human={case.needs_human}")
        if label is not None and case.hypothesis is not None:
            root_score = score_root_cause(case.hypothesis, label)
            lines.append(
                "score_root_cause "
                f"raw_causal_correct={root_score.raw_causal_correct} "
                f"causal_event_correct={root_score.causal_event_correct} "
                f"columns_jaccard={root_score.columns_jaccard}"
            )
        if label is not None and isinstance(clean, pd.DataFrame):
            fix_score = score_fix(case, clean, label)
            lines.append(
                "score_fix "
                f"kind_appropriate={fix_score.kind_appropriate} "
                f"repaired_matches_clean={fix_score.repaired_matches_clean} "
                f"rows_lost_pct={fix_score.rows_lost_pct}"
            )
    lines.append(f"wall_s={wall_s:.2f}")
    return "\n".join(lines)


def _invoke(
    settings: Settings,
    chats: PreparedChats,
    ref: PartitionRef,
    evidence_root: Path,
) -> MeridianState:
    checkpoint = Path(settings.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with (
        shared_ray(settings.ray_num_cpus),
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
