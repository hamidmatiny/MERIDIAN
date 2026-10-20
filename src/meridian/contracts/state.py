"""LangGraph state for one run.

A run holds many incidents. Each incident's response lives in its own
``IncidentCase``, keyed by incident id. Nodes that touch one incident return
that case, and ``merge_cases`` keeps the others. ``append_llm_calls`` records
every model attempt on the run.
"""

from typing import Annotated
from uuid import UUID

from pydantic import Field

from meridian.contracts.base import MeridianModel
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.partition_ref import PartitionRef
from meridian.contracts.tool_call import ToolCall


def merge_cases(
    left: dict[str, IncidentCase],
    right: dict[str, IncidentCase],
) -> dict[str, IncidentCase]:
    """Keep every incident id. The incoming case replaces only its own key."""
    merged = dict(left)
    for key, case in right.items():
        incident_key = str(case.incident.id)
        if str(key) != incident_key:
            raise ValueError("cases key must equal incident.id")
        merged[incident_key] = case
    return merged


def append_llm_calls(left: list[LLMCall], right: list[LLMCall]) -> list[LLMCall]:
    """Append the calls a node just made. Nodes return only their new calls."""
    return [*left, *right]


def append_tool_calls(left: list[ToolCall], right: list[ToolCall]) -> list[ToolCall]:
    """Append the tool calls a node just made. Nodes return only their new calls."""
    return [*left, *right]


class MeridianState(MeridianModel):
    """Checkpointed run state. Loop limits live on each ``IncidentCase``."""

    run_id: UUID = Field(description="Identity of this run. Also the checkpoint thread id.")
    partitions: list[PartitionRef] = Field(
        default_factory=list,
        description="Partitions Sentinel evaluates.",
    )
    cases: Annotated[dict[str, IncidentCase], merge_cases] = Field(
        default_factory=dict,
        description="One case per incident, keyed by incident id.",
    )
    llm_calls: Annotated[list[LLMCall], append_llm_calls] = Field(
        default_factory=list,
        description="Every model attempt made during the run.",
    )
    tool_calls: Annotated[list[ToolCall], append_tool_calls] = Field(
        default_factory=list,
        description="Every tool call made during the run.",
    )
