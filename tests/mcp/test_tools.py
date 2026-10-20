"""Read-only MCP tools return typed payloads and stable refs."""

import hashlib
import json
from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP
from tests.agents.support import DETECTED_AT

from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident import Incident
from meridian.contracts.reference_profile import ColumnProfile, ReferenceProfile
from meridian.contracts.tool_results import (
    ContractSummary,
    EventPage,
    EventView,
    FailureSample,
    LineageResult,
    PartitionDescription,
    ReferenceComparison,
)
from meridian.injector.events import build_events, write_events
from meridian.mcp_servers.client import _tool_text, inprocess_tools, stdio_connections
from meridian.mcp_servers.data import main as data_main
from meridian.mcp_servers.history import main as history_main
from meridian.mcp_servers.impl import (
    compare_to_reference,
    describe_partition,
    evidence_root_from_env,
    get_change_event,
    lineage_result,
    list_change_events,
    sample_failures,
)
from meridian.mcp_servers.lineage import main as lineage_main
from meridian.pipeline.generate import generate_dataset
from meridian.pipeline.parquet import write_parquet


def _snapshot(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def _world(tmp_path: Path) -> tuple[Path, Incident]:
    frame = generate_dataset("vehicle_telemetry", 3, rows=40)
    parquet = tmp_path / "vehicle_telemetry.parquet"
    write_parquet(frame, parquet)
    events, _causal = build_events(
        dataset="vehicle_telemetry",
        seed=3,
        fault=None,
        columns=[],
    )
    write_events(tmp_path / "events.jsonl", events)
    incident = Incident(
        id=events[0].id,
        detected_at=DETECTED_AT,
        dataset="vehicle_telemetry",
        partition="2026-10-19",
        contract_name="VehicleTelemetry",
        fault_type=FaultType.SCHEMA_DRIFT,
        failing_checks=["columns"],
        sample_failure_rows=[{"speed_mps": 1.0}, {"speed_mps": None}],
    )
    dest = tmp_path / "incidents"
    dest.mkdir()
    (dest / f"{incident.id}.json").write_text(incident.model_dump_json(), encoding="utf-8")
    profile = ReferenceProfile(
        dataset="vehicle_telemetry",
        psi_threshold=0.2,
        columns=[
            ColumnProfile(
                column="speed_mps",
                bin_edges=[0.0, 30.0, 60.0],
                expected_proportions=[0.5, 0.5],
            ),
            ColumnProfile(column="missing_col", bin_edges=[0.0, 1.0], expected_proportions=[1.0]),
        ],
    )
    (tmp_path / "profile.json").write_text(profile.model_dump_json(), encoding="utf-8")
    return parquet, incident


def test_each_tool_is_typed_stable_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet, incident = _world(tmp_path)
    monkeypatch.setenv("MERIDIAN_EVIDENCE_ROOT", str(tmp_path))
    before = _snapshot(tmp_path)
    tools = {tool.name: tool for tool in inprocess_tools(tmp_path)}
    lineage = LineageResult.model_validate_json(
        tools["get_upstream"].invoke({"dataset": "trip_summary"})
    )
    again = LineageResult.model_validate_json(
        tools["get_upstream"].invoke({"dataset": "trip_summary"})
    )
    assert lineage.ref == "lineage:upstream:trip_summary"
    assert lineage.datasets == ["vehicle_telemetry"]
    assert again.ref == lineage.ref
    downstream = LineageResult.model_validate_json(
        tools["get_downstream"].invoke({"dataset": "vehicle_telemetry"})
    )
    assert downstream.ref == "lineage:downstream:vehicle_telemetry"
    assert downstream.datasets == ["trip_summary", "fleet_daily_kpi"]
    contract = ContractSummary.model_validate_json(
        tools["get_contract"].invoke({"dataset": "trip_summary"})
    )
    assert contract.ref == "contract:trip_summary"
    assert "avg_speed_mps" in contract.columns
    assert "speed_matches_distance" in contract.checks
    described = PartitionDescription.model_validate_json(
        tools["describe_partition"].invoke({"path": str(parquet)})
    )
    assert described.ref == f"partition:{parquet}"
    assert described.columns[0].ref.startswith(f"column:{parquet}:")
    compared = ReferenceComparison.model_validate_json(
        tools["compare_to_reference"].invoke(
            {"path": str(parquet), "profile_path": str(tmp_path / "profile.json")}
        )
    )
    assert compared.ref == "psi:vehicle_telemetry"
    by_column = {item.column: item for item in compared.columns}
    assert by_column["speed_mps"].ref == "psi:vehicle_telemetry:speed_mps"
    assert by_column["speed_mps"].psi is not None
    assert by_column["missing_col"].psi is None
    assert by_column["missing_col"].null_rate == 1.0
    sample = FailureSample.model_validate_json(
        tools["sample_failures"].invoke({"incident_id": str(incident.id), "n": 1})
    )
    assert sample.ref == f"sample:{incident.id}"
    assert sample.rows == [{"speed_mps": 1.0}]
    page = EventPage.model_validate_json(
        tools["list_events"].invoke({"dataset": "vehicle_telemetry"})
    )
    assert page.ref == "events:vehicle_telemetry"
    assert page.events
    assert page.events[0].ref == f"event:{page.events[0].id}"
    viewed = EventView.model_validate_json(
        tools["get_event"].invoke({"event_id": str(page.events[0].id)})
    )
    assert viewed.ref == page.events[0].ref
    assert _snapshot(tmp_path) == before


def test_history_filters_and_sample_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _parquet, incident = _world(tmp_path)
    monkeypatch.setenv("MERIDIAN_EVIDENCE_ROOT", str(tmp_path))
    page = list_change_events(
        tmp_path,
        dataset="vehicle_telemetry",
        kind="DEPLOY",
        since="2026-10-01T00:00:00+00:00",
        until="2026-10-20T00:00:00+00:00",
    )
    assert all(event.kind.value == "DEPLOY" for event in page.events)
    assert all(event.dataset == "vehicle_telemetry" for event in page.events)
    assert list_change_events(tmp_path, dataset="not-a-dataset").events == []
    assert list_change_events(tmp_path, until="2020-01-01T00:00:00+00:00").events == []
    assert list_change_events(tmp_path, since="2030-01-01T00:00:00+00:00").events == []
    with pytest.raises(ValueError, match="timezone-aware"):
        list_change_events(tmp_path, since="2026-10-01T00:00:00")
    with pytest.raises(KeyError, match="unknown event"):
        get_change_event(tmp_path, "00000000-0000-4000-8000-000000000099")
    with pytest.raises(ValueError, match="1 through 20"):
        sample_failures(tmp_path, str(incident.id), 21)
    with pytest.raises(ValueError, match="1 through 20"):
        sample_failures(tmp_path, str(incident.id), 0)


def test_impl_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = pd.DataFrame({"speed_mps": pd.Series(dtype="float64")})
    path = tmp_path / "empty.parquet"
    empty.to_parquet(path)
    profile = ReferenceProfile(
        dataset="vehicle_telemetry",
        psi_threshold=0.2,
        columns=[
            ColumnProfile(column="speed_mps", bin_edges=[0.0, 1.0], expected_proportions=[1.0]),
        ],
    )
    profile_path = tmp_path / "empty-profile.json"
    profile_path.write_text(profile.model_dump_json(), encoding="utf-8")
    compared = compare_to_reference(str(path), str(profile_path))
    assert compared.columns[0].psi is None
    described = describe_partition(str(path))
    assert described.columns[0].null_rate == 0.0
    with pytest.raises(ValueError, match="direction"):
        lineage_result("vehicle_telemetry", direction="sideways")
    with pytest.raises(KeyError, match="unknown dataset"):
        lineage_result("missing", direction="upstream")
    monkeypatch.delenv("MERIDIAN_EVIDENCE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="MERIDIAN_EVIDENCE_ROOT"):
        evidence_root_from_env()
    mismatched = tmp_path / "incidents" / "not-the-id.json"
    mismatched.parent.mkdir(exist_ok=True)
    incident = Incident(
        id=UUID("11111111-1111-4111-8111-111111111111"),
        detected_at=DETECTED_AT,
        dataset="vehicle_telemetry",
        partition="2026-10-19",
        contract_name="VehicleTelemetry",
        fault_type=FaultType.SCHEMA_DRIFT,
        failing_checks=["columns"],
    )
    mismatched.write_text(incident.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        sample_failures(tmp_path, "not-the-id", 1)


def test_server_mains_use_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def _run(self: FastMCP, transport: str = "stdio") -> None:
        seen.append(f"{self.name}:{transport}")

    monkeypatch.setattr(FastMCP, "run", _run)
    lineage_main()
    data_main()
    history_main()
    assert seen == [
        "meridian-lineage:stdio",
        "meridian-data:stdio",
        "meridian-history:stdio",
    ]


def test_stdio_connection_map_and_payload_shapes(tmp_path: Path) -> None:
    connections = stdio_connections(tmp_path)
    assert set(connections) == {"lineage", "data", "history"}
    assert all(item["transport"] == "stdio" for item in connections.values())
    assert connections["history"]["env"]["MERIDIAN_EVIDENCE_ROOT"] == str(tmp_path)
    assert _tool_text(([], {"ref": "event:1"})) == json.dumps({"ref": "event:1"})
    assert _tool_text({"ref": "events:all"}) == json.dumps({"ref": "events:all"})

    class _Block:
        text = '{"ref": "contract:vehicle_telemetry"}'

    assert json.loads(_tool_text(([_Block()], None)))["ref"].startswith("contract:")
    with pytest.raises(TypeError, match="typed payload"):
        _tool_text(object())


def test_stdio_list_events_round_trip(tmp_path: Path) -> None:
    events, _causal = build_events(
        dataset="fleet_daily_kpi",
        seed=5,
        fault=None,
        columns=[],
    )
    write_events(tmp_path / "events.jsonl", events)
    import asyncio

    from meridian.mcp_servers.client import invoke_tool, load_stdio_tools

    tools = asyncio.run(load_stdio_tools(tmp_path))
    listed = {tool.name: tool for tool in tools}
    payload = invoke_tool(listed["list_events"], {"dataset": "fleet_daily_kpi"})
    page = EventPage.model_validate_json(payload)
    assert page.ref == "events:fleet_daily_kpi"
    assert page.events
    assert all(event.dataset == "fleet_daily_kpi" for event in page.events)
    assert all(event.ref.startswith("event:") for event in page.events)
