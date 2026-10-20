"""Sample rows are capped and older tool payloads are not resent in full."""

from uuid import uuid4

from langchain_core.messages import HumanMessage, ToolMessage
from tests.agents.support import DETECTED_AT

from meridian.contracts.fault_type import FaultType
from meridian.contracts.incident import Incident
from meridian.llm.prompt_size import incident_json, prepare_messages


def test_sample_rows_sent_to_a_model_are_capped() -> None:
    incident = Incident(
        id=uuid4(),
        detected_at=DETECTED_AT,
        dataset="vehicle_telemetry",
        partition="2026-10-19",
        contract_name="VehicleTelemetry",
        fault_type=FaultType.UNIT_MISMATCH,
        failing_checks=["speed"],
        sample_failure_rows=[{"speed_mps": float(index)} for index in range(25)],
    )
    sent, full = incident_json(incident)
    assert sent.count("speed_mps") == 10
    assert full.count("speed_mps") == 25


def test_retries_keep_refs_and_drop_older_payloads() -> None:
    messages = [
        HumanMessage(content="prompt"),
        ToolMessage(
            content='{"ref": "event:abc", "rows": ' + str(list(range(80))) + "}",
            tool_call_id="1",
            name="list_events",
        ),
        ToolMessage(
            content='{"ref": "event:def", "rows": [9]}',
            tool_call_id="2",
            name="get_event",
        ),
    ]
    prepared, before, after = prepare_messages(messages)
    assert before > after
    older = prepared[1]
    assert isinstance(older, ToolMessage)
    assert "event:abc" in str(older.content)
    assert "[1, 2, 3, 4]" not in str(older.content)
    assert "rows" in str(prepared[2].content)
