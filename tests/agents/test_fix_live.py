"""Live Fix → Critic → Verifier. Skipped unless MERIDIAN_LIVE=1."""

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from tests.agents.support import write_partition

from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.live_llm import build_live_chat, build_live_critic
from meridian.bench.score_fix import score
from meridian.contracts.fault_type import FaultType
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.state import MeridianState
from meridian.graph.build import build_graph
from meridian.injector.faults import INJECTORS
from meridian.mcp_servers.client import inprocess_tools
from meridian.pipeline.generate import generate_dataset


def _live(tmp_path: Path, dataset: str, fault: FaultType) -> None:
    if os.environ.get("MERIDIAN_LIVE") != "1":
        pytest.skip("set MERIDIAN_LIVE=1 to call Claude and Grok")
    from meridian.config import get_settings

    settings = get_settings()
    settings.require_anthropic()
    settings.require_xai()
    missing = (
        settings.fix_model is None
        or settings.critic_model is None
        or settings.rootcause_model is None
    )
    if missing:
        pytest.fail("fix, critic, and root-cause model names are required")
    ref = write_partition(
        tmp_path,
        dataset=dataset,
        fault=fault,
        intensity=0.1,
        seed=42,
        rows=80,
        name="live",
    )
    clean = generate_dataset(dataset, 42, rows=80)
    _frame, label = INJECTORS[fault](
        clean,
        np.random.default_rng(42),
        0.1,
        dataset=dataset,
        seed=42,
    )
    from meridian.injector.events import build_events, write_events

    events, _causal = build_events(
        dataset=dataset,
        seed=42,
        fault=fault,
        columns=list(label.columns_affected),
        intensity=0.1,
    )
    write_events(tmp_path / "events.jsonl", events)
    run_id = uuid4()
    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite")) as saver:
        graph = build_graph(
            num_cpus=settings.ray_num_cpus,
            model_name="fake-triage",
            chat=FakeTriageModel(),
            checkpointer=saver,
            root_model_name=settings.rootcause_model,
            root_chat=build_live_chat(
                settings,
                model=settings.rootcause_model,
                model_env="MERIDIAN_ROOTCAUSE_MODEL",
            ),
            evidence_root=tmp_path,
            tools=inprocess_tools(tmp_path),
            max_tool_calls=settings.rootcause_max_tool_calls,
            fix_model_name=settings.fix_model,
            fix_chat=build_live_chat(
                settings,
                model=settings.fix_model,
                model_env="MERIDIAN_FIX_MODEL",
            ),
            critic_model_name=settings.critic_model,
            critic_chat=build_live_critic(settings),
            sandbox_timeout_s=settings.sandbox_timeout_s,
            sandbox_memory_mb=settings.sandbox_memory_mb,
            max_row_loss=settings.max_row_loss,
            max_quarantine=settings.max_quarantine,
        )
        finished = graph.invoke(
            MeridianState(run_id=run_id, partitions=[ref]),
            {"configurable": {"thread_id": str(run_id)}, "recursion_limit": 40},
        )
    state = MeridianState.model_validate(finished)
    case = next(item for item in state.cases.values() if item.incident.fault_type is fault)
    assert case.patch is not None
    scored = score(case, clean, label)
    print(
        json.dumps(
            {
                "fault": fault.value,
                "remediation_kind": case.patch.remediation_kind.value,
                "patch": case.patch.model_dump(mode="json"),
                "verdicts": [item.model_dump(mode="json") for item in case.verdicts],
                "verification": None
                if case.verification is None
                else case.verification.model_dump(mode="json"),
                "score_fix": scored.model_dump(),
                "needs_human": case.needs_human,
            },
            indent=2,
        )
    )
    if fault is FaultType.UNIT_MISMATCH:
        assert case.patch.remediation_kind is RemediationKind.DATA_REPAIR
    else:
        assert case.patch.remediation_kind in {
            RemediationKind.QUARANTINE,
            RemediationKind.ESCALATE,
        }


@pytest.mark.live
def test_live_unit_mismatch_repairs(tmp_path: Path) -> None:
    _live(tmp_path, "vehicle_telemetry", FaultType.UNIT_MISMATCH)


@pytest.mark.live
def test_live_null_explosion_quarantines_or_escalates(tmp_path: Path) -> None:
    _live(tmp_path, "fleet_daily_kpi", FaultType.NULL_EXPLOSION)
