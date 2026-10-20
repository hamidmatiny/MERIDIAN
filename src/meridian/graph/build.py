"""Sentinel fans out to one Triage branch per incident. Later nodes are not wired."""

from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from meridian.agents.sentinel import run_sentinel
from meridian.agents.triage import StructuredChat, run_triage
from meridian.contracts.state import MeridianState
from meridian.prompts import TRIAGE_PROMPT_VERSION


def _as_state(state: object) -> MeridianState:
    """Accept the model LangGraph passes, or the dict a Send branch delivers."""
    if isinstance(state, MeridianState):
        return state
    if isinstance(state, dict):
        return MeridianState.model_validate(state)
    raise TypeError("triage branch input must be MeridianState")


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
            _as_state(state),
            model_name=model_name,
            chat=chat,
            prompt_version=TRIAGE_PROMPT_VERSION,
        )

    graph: StateGraph[MeridianState, None, MeridianState, MeridianState] = StateGraph(MeridianState)
    graph.add_node("sentinel", sentinel)
    graph.add_node("triage", triage)
    graph.add_edge(START, "sentinel")
    graph.add_conditional_edges("sentinel", _fan_out, ["triage", END])
    graph.add_edge("triage", END)
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def _fan_out(state: MeridianState) -> list[Send] | str:
    """Send one triage branch per open case. An empty case map ends the run."""
    pending = [(key, case) for key, case in state.cases.items() if case.triage is None]
    if not pending:
        return END
    return [
        Send(
            "triage",
            {
                "run_id": state.run_id,
                "partitions": list(state.partitions),
                "cases": {key: case},
                "llm_calls": [],
            },
        )
        for key, case in pending
    ]
