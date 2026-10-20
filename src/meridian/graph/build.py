"""Sentinel fans out to triage, then root cause, one branch per incident."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from meridian.agents.root_cause import ToolChat, run_root_cause
from meridian.agents.sentinel import run_sentinel
from meridian.agents.triage import StructuredChat, run_triage
from meridian.contracts.incident_case import IncidentCase
from meridian.contracts.severity import SEVERITY_RANK
from meridian.contracts.state import MeridianState
from meridian.prompts import TRIAGE_PROMPT_VERSION

logger = logging.getLogger("meridian.graph")
ToolLoader = Callable[[], list[BaseTool]]


def _as_state(state: object) -> MeridianState:
    """Accept the model LangGraph passes, or the dict a Send branch delivers."""
    if isinstance(state, MeridianState):
        return state
    if isinstance(state, dict):
        return MeridianState.model_validate(state)
    raise TypeError("branch input must be MeridianState")


def build_graph(
    *,
    num_cpus: int,
    model_name: str,
    chat: StructuredChat,
    checkpointer: Any,
    root_model_name: str,
    root_chat: ToolChat,
    evidence_root: Path,
    tools: list[BaseTool] | None = None,
    tool_loader: ToolLoader | None = None,
    max_tool_calls: int = 8,
    detected_at: datetime | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph[MeridianState, None, MeridianState, MeridianState]:
    """Compile Sentinel → Triage → RootCause. Thread id is the run id."""
    moment = datetime.now(UTC) if detected_at is None else detected_at

    def sentinel(state: MeridianState) -> dict[str, object]:
        return run_sentinel(state, num_cpus=num_cpus, detected_at=moment)

    def triage(state: MeridianState) -> dict[str, object]:
        return run_triage(
            _as_state(state),
            model_name=model_name,
            chat=chat,
            prompt_version=TRIAGE_PROMPT_VERSION,
        )

    def dispatch(state: MeridianState) -> dict[str, object]:
        del state
        return {}

    def root_cause(state: MeridianState) -> dict[str, object]:
        resolved = list(tools) if tools is not None else []
        if tools is None and tool_loader is not None:
            resolved = tool_loader()
        return run_root_cause(
            _as_state(state),
            model_name=root_model_name,
            chat=root_chat,
            tools=resolved,
            max_tool_calls=max_tool_calls,
            evidence_root=evidence_root,
        )

    graph: StateGraph[MeridianState, None, MeridianState, MeridianState] = StateGraph(MeridianState)
    graph.add_node("sentinel", sentinel)
    graph.add_node("triage", triage)
    graph.add_node("dispatch", dispatch, defer=True)
    graph.add_node("root_cause", root_cause)
    graph.add_edge(START, "sentinel")
    graph.add_conditional_edges("sentinel", _fan_out, ["triage", END])
    graph.add_edge("triage", "dispatch")
    graph.add_conditional_edges("dispatch", _to_root, ["root_cause", END])
    graph.add_edge("root_cause", END)
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def _fan_out(state: MeridianState) -> list[Send] | str:
    """Send one triage branch per open case. An empty case map ends the run."""
    pending = [(key, case) for key, case in state.cases.items() if case.triage is None]
    if not pending:
        return END
    return [Send("triage", _branch(state, key, case)) for key, case in pending]


def _to_root(state: MeridianState) -> list[Send] | str:
    """Log severity order, then send one root-cause branch per case. Branches stay parallel."""
    ranked = sorted(state.cases.items(), key=_severity_key)
    if not ranked:
        return END
    order = ", ".join(
        f"{key}:{case.severity.value if case.severity is not None else 'unset'}"
        for key, case in ranked
    )
    logger.info("root cause order: %s", order)
    return [Send("root_cause", _branch(state, key, case)) for key, case in ranked]


def _branch(state: MeridianState, key: str, case: IncidentCase) -> dict[str, object]:
    return {
        "run_id": state.run_id,
        "partitions": list(state.partitions),
        "cases": {key: case},
        "llm_calls": [],
        "tool_calls": [],
    }


def _severity_key(item: tuple[str, IncidentCase]) -> tuple[int, str]:
    severity = item[1].severity
    rank = SEVERITY_RANK[severity] if severity is not None else 0
    return (-rank, item[0])
