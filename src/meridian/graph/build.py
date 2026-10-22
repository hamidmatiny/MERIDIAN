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
from meridian.contracts.remediation import RemediationKind
from meridian.contracts.severity import SEVERITY_RANK
from meridian.contracts.state import MeridianState
from meridian.graph.limits import route_after_root
from meridian.llm.errors import AbortGate, RunAborted, unwrap_abort
from meridian.progress import node_line
from meridian.prompts import TRIAGE_PROMPT_VERSION
from meridian.slack.gate import ApprovalPorts, run_approval, run_postmortem
from meridian.wall import WallClock, call_bounded

logger = logging.getLogger("meridian.graph")

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


def _case_label(state: MeridianState) -> str:
    if len(state.cases) != 1:
        return "-"
    return next(iter(state.cases))


def _attempt_no(node: str, state: MeridianState) -> int:
    if len(state.cases) != 1:
        return 1
    case = next(iter(state.cases.values()))
    if node == "fix":
        return case.fix_attempts + 1
    if node == "critic":
        return case.critic_rounds + 1
    return 1


def _trace(
    clock: WallClock,
    node: str,
    state: MeridianState,
    body: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Log the node, then run it. A wall abort still gets an end line."""
    case = _case_label(state)
    attempt = _attempt_no(node, state)
    node_line(state.run_id, case, node, attempt, clock.elapsed(), "start")
    try:
        result = body()
    except Exception as exc:
        if type(exc).__name__ != "GraphInterrupt":
            node_line(state.run_id, case, node, attempt, clock.elapsed(), "end")
        raise
    node_line(state.run_id, case, node, attempt, clock.elapsed(), "end")
    return result


def _limited(clock: WallClock, action: Callable[[], dict[str, object]]) -> dict[str, object]:
    """Poll the wall clock so a blocked node is abandoned."""
    return call_bounded(action, clock)


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
    max_outside_change: float = 0.0,
    sample_seed: int = 42,
    wall_budget_s: float = 900,
    detected_at: datetime | None = None,
    interrupt_after: list[str] | None = None,
    psi_min_rows: int = 200,
    approval: bool = False,
    approval_ports: ApprovalPorts | None = None,
    wall_clock: WallClock | None = None,
) -> CompiledStateGraph[MeridianState, None, MeridianState, MeridianState]:
    """Compile Sentinel → Triage → RootCause → Fix → Critic → Verifier.

    ``approval=True`` continues a finished branch into the Slack gate and a
    postmortem. The gate interrupts; it does not block the process.
    """
    if approval and approval_ports is None:
        raise ValueError("approval requires approval_ports")
    moment = datetime.now(UTC) if detected_at is None else detected_at
    fix_model = HeuristicFixModel()
    critic_model = HeuristicCriticModel()
    chosen_fix = fix_model if fix_chat is None else fix_chat
    chosen_fix_name = fix_model.model_name if fix_model_name is None else fix_model_name
    chosen_critic = critic_model if critic_chat is None else critic_chat
    chosen_critic_name = critic_model.model_name if critic_model_name is None else critic_model_name
    gate = AbortGate()
    clock = wall_clock if wall_clock is not None else WallClock(wall_budget_s)

    def _over() -> bool:
        return clock.expired()

    def _human(state: MeridianState) -> dict[str, object]:
        cases = {
            key: case.model_copy(update={"needs_human": True}) for key, case in state.cases.items()
        }
        return {"aborted": True, "abort_reason": "WALL_BUDGET", "cases": cases}

    def sentinel(state: MeridianState) -> dict[str, object]:
        def node(current: MeridianState) -> dict[str, object]:
            def body() -> dict[str, object]:
                if clock.expired():
                    return run_sentinel(current, num_cpus=num_cpus, detected_at=moment)
                return _limited(
                    clock,
                    lambda: run_sentinel(current, num_cpus=num_cpus, detected_at=moment),
                )

            return _trace(clock, "sentinel", current, body)

        return _guard(gate, state, node, human=_human)

    def triage(state: MeridianState) -> dict[str, object]:
        def node(current: MeridianState) -> dict[str, object]:
            return _trace(
                clock,
                "triage",
                current,
                lambda: _limited(
                    clock,
                    lambda: run_triage(
                        current,
                        model_name=model_name,
                        chat=chat,
                        prompt_version=TRIAGE_PROMPT_VERSION,
                    ),
                ),
            )

        return _guard(gate, state, node, over=_over, human=_human)

    def dispatch(state: MeridianState) -> dict[str, object]:
        del state
        return {}

    def root_cause(state: MeridianState) -> dict[str, object]:
        resolved = list(tools) if tools is not None else []
        if tools is None and tool_loader is not None:
            resolved = tool_loader()

        def _run(current: MeridianState) -> dict[str, object]:
            return _trace(
                clock,
                "root_cause",
                current,
                lambda: _limited(
                    clock,
                    lambda: run_root_cause(
                        current,
                        model_name=root_model_name,
                        chat=root_chat,
                        tools=resolved,
                        max_tool_calls=max_tool_calls,
                        evidence_root=evidence_root,
                    ),
                ),
            )

        return _guard(gate, state, _run, over=_over, human=_human)

    def fix(state: MeridianState) -> dict[str, object]:
        resolved = list(tools) if tools is not None else []
        if tools is None and tool_loader is not None:
            resolved = tool_loader()

        def _run(current: MeridianState) -> dict[str, object]:
            return _trace(
                clock,
                "fix",
                current,
                lambda: _limited(
                    clock,
                    lambda: run_fix(
                        current,
                        model_name=chosen_fix_name,
                        chat=chosen_fix,
                        tools=resolved,
                        max_tool_calls=max_tool_calls,
                        max_row_loss=max_row_loss,
                        max_quarantine=max_quarantine,
                    ),
                ),
            )

        return _guard(gate, state, _run, over=_over, human=_human)

    def pre_critic(state: MeridianState) -> dict[str, object]:
        def node(current: MeridianState) -> dict[str, object]:
            return _trace(
                clock,
                "pre_critic",
                current,
                lambda: _limited(
                    clock,
                    lambda: run_pre_critic(
                        current,
                        num_cpus=num_cpus,
                        timeout_s=sandbox_timeout_s,
                        memory_mb=sandbox_memory_mb,
                        sample_seed=sample_seed,
                        psi_min_rows=psi_min_rows,
                    ),
                ),
            )

        return _guard(gate, state, node, over=_over, human=_human)

    def critic(state: MeridianState) -> dict[str, object]:
        def node(current: MeridianState) -> dict[str, object]:
            return _trace(
                clock,
                "critic",
                current,
                lambda: _limited(
                    clock,
                    lambda: run_critic(
                        current,
                        model_name=chosen_critic_name,
                        chat=chosen_critic,
                    ),
                ),
            )

        return _guard(gate, state, node, over=_over, human=_human)

    def verifier(state: MeridianState) -> dict[str, object]:
        def node(current: MeridianState) -> dict[str, object]:
            return _trace(
                clock,
                "verifier",
                current,
                lambda: _limited(
                    clock,
                    lambda: run_verifier(
                        current,
                        num_cpus=num_cpus,
                        timeout_s=sandbox_timeout_s,
                        memory_mb=sandbox_memory_mb,
                        max_row_loss=max_row_loss,
                        max_quarantine=max_quarantine,
                        max_outside_change=max_outside_change,
                        psi_min_rows=psi_min_rows,
                    ),
                ),
            )

        return _guard(gate, state, node, over=_over, human=_human)

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
    if approval:
        assert approval_ports is not None
        ports = approval_ports

        def slack_approval(state: MeridianState) -> dict[str, object]:
            current = _as_state(state)
            return _trace(clock, "slack_approval", current, lambda: run_approval(current, ports))

        def postmortem(state: MeridianState) -> dict[str, object]:
            current = _as_state(state)
            return _trace(clock, "postmortem", current, lambda: run_postmortem(current, ports))

        graph.add_node("slack_approval", slack_approval)
        graph.add_node("postmortem", postmortem)
        graph.add_node("route_slack", _wait, defer=True)
        graph.add_edge("slack_approval", "route_slack")
        graph.add_conditional_edges("route_slack", _to_postmortem, ["postmortem", END])
        graph.add_edge("postmortem", END)
    tail: list[str] = ["slack_approval", END] if approval else [END]
    graph.add_edge(START, "sentinel")
    graph.add_conditional_edges("sentinel", _fan_out, ["triage", END])
    graph.add_edge("triage", "dispatch")
    graph.add_conditional_edges("dispatch", _with_slack(_to_root, approval), ["root_cause", *tail])
    graph.add_edge("root_cause", "route_root")
    graph.add_conditional_edges("route_root", _with_slack(_after_root, approval), ["fix", *tail])
    graph.add_edge("fix", "route_fix")
    graph.add_conditional_edges(
        "route_fix",
        _with_slack(_to_pre_critic, approval),
        ["pre_critic", *tail],
    )
    graph.add_edge("pre_critic", "route_pre")
    graph.add_conditional_edges(
        "route_pre",
        _with_slack(_after_pre_critic, approval),
        ["fix", "critic", *tail],
    )
    graph.add_edge("critic", "route_critic")
    graph.add_conditional_edges(
        "route_critic",
        _with_slack(_after_critic, approval),
        ["fix", "critic", "verifier", *tail],
    )
    graph.add_edge("verifier", "route_verifier")
    graph.add_conditional_edges(
        "route_verifier",
        _with_slack(_after_verifier, approval),
        ["fix", *tail],
    )
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


def _escalate_open(case: IncidentCase) -> bool:
    """An ESCALATE patch still walks to the critic until the verifier records it."""
    return (
        case.patch is not None
        and case.patch.remediation_kind is RemediationKind.ESCALATE
        and case.verification is None
    )


def _to_postmortem(state: MeridianState) -> list[Send] | str:
    """One postmortem branch per incident. A joined wall abort has more than one."""
    pending = [(key, case) for key, case in state.cases.items() if case.postmortem is None]
    if not pending:
        return END
    return [Send("postmortem", _branch(state, key, case)) for key, case in pending]


def _with_slack(
    router: Callable[[MeridianState], list[Send] | str],
    enabled: bool,
) -> Callable[[MeridianState], list[Send] | str]:
    """Turn a finished branch into the Slack gate when approval is on."""

    def inner(state: MeridianState) -> list[Send] | str:
        result = router(state)
        if result != END or not enabled:
            return result
        pending = [(key, case) for key, case in state.cases.items() if case.postmortem is None]
        if not pending:
            return END
        return [Send("slack_approval", _branch(state, key, case)) for key, case in pending]

    return inner


def _to_pre_critic(state: MeridianState) -> list[Send] | str:
    """Send a patch to the deterministic checks before any critic call."""
    if state.aborted:
        return END
    pending = [
        (key, case)
        for key, case in state.cases.items()
        if case.patch is not None
        and case.pre_critic_attempt < case.fix_attempts
        and (not case.needs_human or _escalate_open(case))
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
        if case.patch is None:
            continue
        if case.needs_human and not _escalate_open(case):
            continue
        if (
            case.pre_critic_blocked
            and not case.needs_human
            and case.fix_attempts < MAX_FIX_ATTEMPTS
        ):
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
        if case.patch is None or not case.verdicts:
            continue
        if case.needs_human and not _escalate_open(case):
            continue
        latest = case.verdicts[-1]
        degraded = sum(1 for item in case.verdicts if item.critic_degraded)
        if latest.critic_degraded and degraded < 2 and not case.needs_human:
            sends.append(Send("critic", _branch(state, key, case)))
        elif latest.approved and case.verification is None:
            sends.append(Send("verifier", _branch(state, key, case)))
        elif (
            not latest.approved
            and not latest.critic_degraded
            and not case.needs_human
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
