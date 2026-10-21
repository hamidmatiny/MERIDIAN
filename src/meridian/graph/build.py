"""Sentinel fans out to triage, then root cause, fix, critic, and verifier."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from meridian.agents.critic import CriticChat, run_critic
from meridian.agents.fake_critic import HeuristicCriticModel
from meridian.agents.fake_fix import HeuristicFixModel
from meridian.agents.fix import run_fix
from meridian.agents.pre_critic import run_pre_critic
from meridian.agents.root_cause import ToolChat, run_root_cause
from meridian.agents.sentinel import run_sentinel
from meridian.agents.triage import StructuredChat, run_triage
from meridian.agents.verifier import run_verifier
from meridian.contracts.incident_case import MAX_CRITIC_ROUNDS, MAX_FIX_ATTEMPTS, IncidentCase
from meridian.contracts.severity import SEVERITY_RANK
from meridian.contracts.state import MeridianState
from meridian.graph.limits import route_after_root
from meridian.llm.errors import AbortGate, RunAborted, unwrap_abort
from meridian.prompts import TRIAGE_PROMPT_VERSION

logger = logging.getLogger("meridian.graph")


def _clock() -> float:
    import time

    return time.monotonic()


ToolLoader = Callable[[], list[BaseTool]]


def _guard(
    gate: AbortGate,
    state: object,
    node: Callable[[MeridianState], dict[str, object]],
    *,
    over: Callable[[], bool] | None = None,
    human: Callable[[MeridianState], dict[str, object]] | None = None,
) -> dict[str, object]:
    """Skip the model once the run has aborted, and record a fatal provider error."""
    current = _as_state(state)
    if gate.reason is not None:
        if gate.reason == "WALL_BUDGET" and human is not None:
            return human(current)
        return {"aborted": True, "abort_reason": gate.reason}
    if current.aborted:
        reason = current.abort_reason or "aborted"
        gate.trip(reason)
        return {"aborted": True, "abort_reason": reason}
    if over is not None and over():
        return _wall(gate, current, human)
    try:
        update = node(current)
    except Exception as exc:
        aborted = exc if isinstance(exc, RunAborted) else unwrap_abort(exc)
        if aborted is None:
            raise
        if aborted.kind == "WALL_BUDGET":
            return _wall(gate, current, human)
        gate.trip(str(aborted))
        return {"aborted": True, "abort_reason": str(aborted)}
    raw_reason = update.get("abort_reason")
    if update.get("aborted"):
        text = raw_reason if isinstance(raw_reason, str) and raw_reason else "aborted"
        gate.trip(text)
    return update


def _wall(
    gate: AbortGate,
    state: MeridianState,
    human: Callable[[MeridianState], dict[str, object]] | None,
) -> dict[str, object]:
    gate.trip("WALL_BUDGET")
    if human is not None:
        return human(state)
    return {"aborted": True, "abort_reason": "WALL_BUDGET"}


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
    fix_model_name: str | None = None,
    fix_chat: ToolChat | None = None,
    critic_model_name: str | None = None,
    critic_chat: CriticChat | None = None,
    sandbox_timeout_s: float = 20,
    sandbox_memory_mb: int = 512,
    max_row_loss: float = 0.01,
    max_quarantine: float = 0.25,
    sample_seed: int = 42,
    wall_budget_s: float = 900,
    detected_at: datetime | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledStateGraph[MeridianState, None, MeridianState, MeridianState]:
    """Compile Sentinel → Triage → RootCause → Fix → Critic → Verifier."""
    moment = datetime.now(UTC) if detected_at is None else detected_at
    fix_model = HeuristicFixModel()
    critic_model = HeuristicCriticModel()
    chosen_fix = fix_model if fix_chat is None else fix_chat
    chosen_fix_name = fix_model.model_name if fix_model_name is None else fix_model_name
    chosen_critic = critic_model if critic_chat is None else critic_chat
    chosen_critic_name = critic_model.model_name if critic_model_name is None else critic_model_name
    gate = AbortGate()
    started = _clock()

    def _over() -> bool:
        return _clock() - started >= wall_budget_s

    def _human(state: MeridianState) -> dict[str, object]:
        cases = {
            key: case.model_copy(update={"needs_human": True}) for key, case in state.cases.items()
        }
        return {"aborted": True, "abort_reason": "WALL_BUDGET", "cases": cases}

    def sentinel(state: MeridianState) -> dict[str, object]:
        return run_sentinel(state, num_cpus=num_cpus, detected_at=moment)

    def triage(state: MeridianState) -> dict[str, object]:
        return _guard(
            gate,
            state,
            lambda current: run_triage(
                current,
                model_name=model_name,
                chat=chat,
                prompt_version=TRIAGE_PROMPT_VERSION,
            ),
            over=_over,
            human=_human,
        )

    def dispatch(state: MeridianState) -> dict[str, object]:
        del state
        return {}

    def root_cause(state: MeridianState) -> dict[str, object]:
        resolved = list(tools) if tools is not None else []
        if tools is None and tool_loader is not None:
            resolved = tool_loader()

        def _run(current: MeridianState) -> dict[str, object]:
            return run_root_cause(
                current,
                model_name=root_model_name,
                chat=root_chat,
                tools=resolved,
                max_tool_calls=max_tool_calls,
                evidence_root=evidence_root,
            )

        return _guard(gate, state, _run, over=_over, human=_human)

    def fix(state: MeridianState) -> dict[str, object]:
        resolved = list(tools) if tools is not None else []
        if tools is None and tool_loader is not None:
            resolved = tool_loader()

        def _run(current: MeridianState) -> dict[str, object]:
            return run_fix(
                current,
                model_name=chosen_fix_name,
                chat=chosen_fix,
                tools=resolved,
                max_tool_calls=max_tool_calls,
                max_row_loss=max_row_loss,
                max_quarantine=max_quarantine,
            )

        return _guard(gate, state, _run, over=_over, human=_human)

    def pre_critic(state: MeridianState) -> dict[str, object]:
        current = _as_state(state)
        if _over():
            return _wall(gate, current, _human)
        if gate.reason is not None or current.aborted:
            reason = gate.reason or current.abort_reason or "aborted"
            gate.trip(reason)
            return {"aborted": True, "abort_reason": reason}
        return run_pre_critic(
            current,
            num_cpus=num_cpus,
            timeout_s=sandbox_timeout_s,
            memory_mb=sandbox_memory_mb,
            sample_seed=sample_seed,
        )

    def critic(state: MeridianState) -> dict[str, object]:
        return _guard(
            gate,
            state,
            lambda current: run_critic(
                current,
                model_name=chosen_critic_name,
                chat=chosen_critic,
            ),
            over=_over,
            human=_human,
        )

    def verifier(state: MeridianState) -> dict[str, object]:
        current = _as_state(state)
        if _over():
            return _wall(gate, current, _human)
        return run_verifier(
            current,
            num_cpus=num_cpus,
            timeout_s=sandbox_timeout_s,
            memory_mb=sandbox_memory_mb,
            max_row_loss=max_row_loss,
            max_quarantine=max_quarantine,
        )

    graph: StateGraph[MeridianState, None, MeridianState, MeridianState] = StateGraph(MeridianState)
    graph.add_node("sentinel", sentinel)
    graph.add_node("triage", triage)
    graph.add_node("dispatch", dispatch, defer=True)
    graph.add_node("root_cause", root_cause)
    graph.add_node("fix", fix)
    graph.add_node("pre_critic", pre_critic)
    graph.add_node("critic", critic)
    graph.add_node("verifier", verifier)
    graph.add_node("route_root", _wait, defer=True)
    graph.add_node("route_fix", _wait, defer=True)
    graph.add_node("route_pre", _wait, defer=True)
    graph.add_node("route_critic", _wait, defer=True)
    graph.add_node("route_verifier", _wait, defer=True)
    graph.add_edge(START, "sentinel")
    graph.add_conditional_edges("sentinel", _fan_out, ["triage", END])
    graph.add_edge("triage", "dispatch")
    graph.add_conditional_edges("dispatch", _to_root, ["root_cause", END])
    graph.add_edge("root_cause", "route_root")
    graph.add_conditional_edges("route_root", _after_root, ["fix", END])
    graph.add_edge("fix", "route_fix")
    graph.add_conditional_edges("route_fix", _to_pre_critic, ["pre_critic", END])
    graph.add_edge("pre_critic", "route_pre")
    graph.add_conditional_edges("route_pre", _after_pre_critic, ["fix", "critic", END])
    graph.add_edge("critic", "route_critic")
    graph.add_conditional_edges("route_critic", _after_critic, ["fix", "critic", "verifier", END])
    graph.add_edge("verifier", "route_verifier")
    graph.add_conditional_edges("route_verifier", _after_verifier, ["fix", END])
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def _fan_out(state: MeridianState) -> list[Send] | str:
    """Send one triage branch per open case. An empty case map ends the run."""
    if state.aborted:
        return END
    pending = [(key, case) for key, case in state.cases.items() if case.triage is None]
    if not pending:
        return END
    return [Send("triage", _branch(state, key, case)) for key, case in pending]


def _to_root(state: MeridianState) -> list[Send] | str:
    """Log severity order, then send one root-cause branch per case. Branches stay parallel."""
    if state.aborted:
        return END
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
        "aborted": state.aborted,
        "abort_reason": state.abort_reason,
    }


def _severity_key(item: tuple[str, IncidentCase]) -> tuple[int, str]:
    severity = item[1].severity
    rank = SEVERITY_RANK[severity] if severity is not None else 0
    return (-rank, item[0])


def _wait(state: MeridianState) -> dict[str, object]:
    """Join parallel branches. The conditional edge after this node fans out again."""
    del state
    return {}


def _after_root(state: MeridianState) -> list[Send] | str:
    """Send each case that has a hypothesis and no patch yet to its own Fix branch."""
    if state.aborted:
        return END
    pending = [
        (key, case)
        for key, case in state.cases.items()
        if route_after_root(case) == "fix" and case.fix_attempts == 0
    ]
    if not pending:
        return END
    return [Send("fix", _branch(state, key, case)) for key, case in pending]


def _to_pre_critic(state: MeridianState) -> list[Send] | str:
    """Send a patch to the deterministic checks before any critic call."""
    if state.aborted:
        return END
    pending = [
        (key, case)
        for key, case in state.cases.items()
        if case.patch is not None
        and not case.needs_human
        and case.pre_critic_attempt < case.fix_attempts
    ]
    if not pending:
        return END
    return [Send("pre_critic", _branch(state, key, case)) for key, case in pending]


def _after_pre_critic(state: MeridianState) -> list[Send] | str:
    """A syntax or gate failure returns to Fix. A passing dry-run goes to the critic."""
    if state.aborted:
        return END
    sends: list[Send] = []
    for key, case in state.cases.items():
        if case.needs_human or case.patch is None:
            continue
        if case.pre_critic_blocked and case.fix_attempts < MAX_FIX_ATTEMPTS:
            sends.append(Send("fix", _branch(state, key, case)))
        elif (
            not case.pre_critic_blocked
            and case.pre_critic_attempt == case.fix_attempts
            and len(case.verdicts) < case.fix_attempts
        ):
            sends.append(Send("critic", _branch(state, key, case)))
    return sends or END


def _after_critic(state: MeridianState) -> list[Send] | str:
    """Approve goes to Verifier. A degraded review retries the critic, not Fix."""
    if state.aborted:
        return END
    sends: list[Send] = []
    for key, case in state.cases.items():
        if case.needs_human or not case.verdicts or case.patch is None:
            continue
        latest = case.verdicts[-1]
        degraded = sum(1 for item in case.verdicts if item.critic_degraded)
        if latest.critic_degraded and degraded < 2:
            sends.append(Send("critic", _branch(state, key, case)))
        elif latest.approved and case.verification is None:
            sends.append(Send("verifier", _branch(state, key, case)))
        elif (
            not latest.approved
            and not latest.critic_degraded
            and case.fix_attempts == case.critic_rounds
            and case.critic_rounds < MAX_CRITIC_ROUNDS
        ):
            sends.append(Send("fix", _branch(state, key, case)))
    return sends or END


def _after_verifier(state: MeridianState) -> list[Send] | str:
    """A failed replay under the cap goes back to Fix. Anything else ends."""
    if state.aborted:
        return END
    pending = [
        (key, case)
        for key, case in state.cases.items()
        if case.verification is not None
        and not case.verification.passed
        and not case.needs_human
        and case.fix_attempts < MAX_FIX_ATTEMPTS
    ]
    if not pending:
        return END
    return [Send("fix", _branch(state, key, case)) for key, case in pending]
