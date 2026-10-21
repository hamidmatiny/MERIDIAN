"""Live Claude root cause on one schema-drift case. Skipped unless MERIDIAN_LIVE=1."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from tests.agents.support import DETECTED_AT, write_partition

from meridian.agents.root_cause import run_root_cause
from meridian.bench.score_root_cause import score
from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.reference_profile import ReferenceProfile
from meridian.contracts.severity import Severity
from meridian.contracts.state import MeridianState
from meridian.contracts.triage import TriageResult
from meridian.injector.events import build_events, require_causal_event, write_events
from meridian.injector.faults import INJECTORS
from meridian.mcp_servers.client import inprocess_tools
from meridian.pipeline.generate import generate_dataset
from meridian.pipeline.parquet import read_parquet
from meridian.quality.evaluate import evaluate


@pytest.mark.live
def test_live_root_cause_on_schema_drift(tmp_path: Path) -> None:
    if os.environ.get("MERIDIAN_LIVE") != "1":
        pytest.skip("set MERIDIAN_LIVE=1 to call the configured model")
    from uuid import uuid4

    from meridian.config import get_settings
    from meridian.llm.router import build_node_chat

    settings = get_settings()
    from meridian.llm.resident import budget_problem, read_ollama_sizes

    problem = budget_problem(settings, read_ollama_sizes())
    if problem is not None:
        pytest.fail(problem)
    if settings.rootcause_provider in {"anthropic", "xai"} and not settings.allow_paid:
        pytest.skip("paid provider requires MERIDIAN_ALLOW_PAID=1")
    chat, model_name = build_node_chat(settings, "rootcause")
    from meridian.llm.capabilities import arm_node

    arm_node(settings, "rootcause", chat)
    ref = write_partition(
        tmp_path,
        dataset="vehicle_telemetry",
        fault=FaultType.SCHEMA_DRIFT,
        intensity=0.3,
        seed=42,
        rows=80,
        name="drift",
    )
    frame = generate_dataset("vehicle_telemetry", 42, rows=80)
    _injected, label = INJECTORS[FaultType.SCHEMA_DRIFT](
        frame,
        np.random.default_rng(42),
        0.3,
        dataset="vehicle_telemetry",
        seed=42,
    )
    events, causal = build_events(
        dataset="vehicle_telemetry",
        seed=42,
        fault=FaultType.SCHEMA_DRIFT,
        columns=list(label.columns_affected),
        intensity=0.3,
    )
    assert causal == label.causal_event_id
    write_events(tmp_path / "events.jsonl", events)
    require_causal_event(label, tmp_path / "events.jsonl")
    profile = ReferenceProfile.model_validate_json(
        Path(ref.reference_profile_path).read_text(encoding="utf-8")
    )
    incidents = evaluate(
        ref.dataset,
        read_parquet(Path(ref.parquet_path)),
        ref.metadata,
        profile,
        detected_at=DETECTED_AT,
    )
    incident = next(item for item in incidents if item.fault_type is FaultType.SCHEMA_DRIFT)
    case = IncidentCase(
        incident=incident,
        triage=TriageResult(
            incident_id=incident.id,
            severity=Severity.SEV2,
            rationale="Floor kept this incident at SEV2.",
            affected_consumers=list(incident.blast_radius),
            confidence=0.4,
            llm_severity=Severity.SEV2,
            floor_severity=Severity.SEV2,
            final_severity=Severity.SEV2,
            prompt_version="v1",
        ),
    )
    updated = run_root_cause(
        MeridianState(run_id=uuid4(), cases={str(incident.id): case}),
        model_name=model_name,
        chat=chat,
        tools=inprocess_tools(tmp_path),
        max_tool_calls=settings.rootcause_max_tool_calls,
        evidence_root=tmp_path,
    )
    hypothesis = updated["cases"][str(incident.id)].hypothesis
    assert hypothesis is not None
    scored = score(hypothesis, label)
    for call in updated["llm_calls"]:
        usd = "0.00" if call.provider not in {"anthropic", "xai"} else "paid"
        print(
            f"provider={call.provider} model={call.model} "
            f"input_tokens={call.input_tokens} output_tokens={call.output_tokens} usd={usd}"
        )
    print(
        json.dumps(
            {
                "hypothesis": hypothesis.model_dump(mode="json"),
                "score": scored.model_dump(),
                "causal_event_correct": scored.causal_event_correct,
            },
            indent=2,
        )
    )
