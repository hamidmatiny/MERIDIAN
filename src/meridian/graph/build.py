"""Sentinel → Triage. Later nodes are not wired."""

from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from meridian.agents.sentinel import run_sentinel
from meridian.agents.triage import StructuredChat, run_triage
from meridian.contracts.state import MeridianState
from meridian.prompts import TRIAGE_PROMPT_VERSION


def build_graph(
    *,
    num_cpus: int,
    model_name: str,
    chat: StructuredChat,
    checkpointer: Any,
    detected_at: datetime | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph[MeridianState, None, MeridianState, MeridianState]:
    """Compile the Phase 2 graph. Thread id is the run id, supplied by the caller."""
    moment = datetime.now(UTC) if detected_at is None else detected_at

    def sentinel(state: MeridianState) -> dict[str, object]:
        return run_sentinel(state, num_cpus=num_cpus, detected_at=moment)

    def triage(state: MeridianState) -> dict[str, object]:
        return run_triage(
            state,
            model_name=model_name,
            chat=chat,
            prompt_version=TRIAGE_PROMPT_VERSION,
        )

    graph: StateGraph[MeridianState, None, MeridianState, MeridianState] = StateGraph(MeridianState)
    graph.add_node("sentinel", sentinel)
    graph.add_node("triage", triage)
    graph.add_edge(START, "sentinel")
    graph.add_conditional_edges("sentinel", _after_sentinel)
    graph.add_edge("triage", END)
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def _after_sentinel(state: MeridianState) -> str:
    if state.incidents:
        return "triage"
    return END
