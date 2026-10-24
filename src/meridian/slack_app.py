"""Socket Mode handler and the graph worker, in one process.

Run ``python -m meridian.slack_app``. Button clicks resume a checkpoint.
"""

import argparse
import logging
import signal
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from meridian.agents.fake_llm import FakeTriageModel
from meridian.agents.fake_root_cause import HeuristicRootCauseModel
from meridian.config import MeridianConfigError, Settings, get_settings
from meridian.graph.build import build_graph
from meridian.slack.handlers import (
    COMMENT_REQUIRED,
    expire_cases,
    on_mention,
    on_reject_submit,
)
from meridian.slack.probe import (
    SlackIdentity,
    connections_open,
    fetch_identity,
    startup_problem,
)
from meridian.slack.process import shutdown_process
from meridian.slack.runtime import ports_for
from meridian.slack.store import DecisionLog

logger = logging.getLogger("meridian.slack")


def configure_logging(path: Path) -> None:
    """Log to stderr and to ``path``. Tokens are not written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path)],
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Probe Slack, then run the socket and the graph worker until a signal."""
    parser = argparse.ArgumentParser(prog="meridian.slack_app")
    parser.add_argument("--log", default="data/slack_app.log")
    args = parser.parse_args(argv)
    configure_logging(Path(args.log))
    from meridian.audit.store import install_from_env
    from meridian.tracing import enabled as tracing_enabled

    settings = get_settings()
    install_from_env()
    tracing_enabled()
    try:
        settings.require_slack()
    except MeridianConfigError as exc:
        logger.error("%s", exc)
        return 2
    channel = settings.slack_approval_channel or ""
    bot_token = _secret(settings.slack_bot_token)
    app_token = _secret(settings.slack_app_token)
    identity = fetch_identity(WebClient(token=bot_token), channel)
    opened, error = connections_open(WebClient(token=app_token))
    if not opened:
        logger.error("apps.connections.open error=%s", error or "not ok")
        return 2
    problem = startup_problem(identity, settings.approver_ids())
    if problem is not None:
        logger.error("%s", problem)
        return 2
    logger.info(
        "bot_user_id=%s bot_name=%s team=%s channel=%s member=%s",
        identity.bot_user_id,
        identity.bot_name,
        identity.team,
        identity.channel_name,
        identity.member,
    )
    return serve(settings, identity)


def serve(settings: Settings, identity: SlackIdentity) -> int:
    """Block on Socket Mode. SIGINT and SIGTERM close Ray and the socket."""
    checkpoint = Path(settings.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ports = ports_for(settings)
    decisions = DecisionLog(Path("data/slack-decisions"))
    stop = threading.Event()
    lock = threading.Lock()
    sessions: list[Any] = []
    with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
        graph = _worker(settings, saver, ports)
        app = App(
            token=_secret(settings.slack_bot_token),
            token_verification_enabled=False,
        )

        def resume(thread_id: str, payload: dict[str, object]) -> None:
            with lock:
                graph.invoke(
                    Command(resume=payload),
                    {"configurable": {"thread_id": thread_id}},
                )

        register(
            app,
            bot_user_id=identity.bot_user_id,
            approvers=settings.approver_ids(),
            decisions=decisions,
            cases=ports.open_index.cases,
            resume=resume,
            timeout_s=settings.approval_timeout_s,
            audit=ports.audit,
        )

        def watch() -> None:
            while not stop.wait(5):
                expire_cases(
                    cases=ports.open_index.cases(),
                    timeout_s=settings.approval_timeout_s,
                    now=datetime.now(UTC),
                    log=decisions,
                    resume=resume,
                )

        thread = threading.Thread(target=watch, name="meridian-approval-timeout", daemon=True)
        thread.start()
        handler = SocketModeHandler(app, _secret(settings.slack_app_token))

        def stop_all() -> None:
            stop.set()
            shutdown_process(
                close_socket=handler.close,
                sessions=sessions,
                ray_module=_ray(),
            )

        _install(stop_all)
        try:
            handler.start()  # type: ignore[no-untyped-call]
        finally:
            stop.set()
    return 0


def register(
    app: App,
    *,
    bot_user_id: str,
    approvers: frozenset[str],
    decisions: DecisionLog,
    cases: Callable[[], list[Any]],
    resume: Callable[[str, dict[str, object]], None],
    timeout_s: int,
    audit: Any | None = None,
) -> None:
    """Attach mention, button, and modal handlers. ``timeout_s`` is logged."""
    logger.info("approval_timeout_s=%s", timeout_s)

    def mention(event: dict[str, Any], say: Callable[..., None]) -> None:
        on_mention(
            event,
            say,
            bot_user_id=bot_user_id,
            cases=list(cases()),
            now=datetime.now(UTC),
            audit=audit,
        )

    def action(ack: Callable[..., None], body: dict[str, Any], client: Any) -> None:
        ack()
        note = _action(body, approvers=approvers, decisions=decisions, resume=resume, audit=audit)
        _ephemeral_or_modal(body, client, note)

    def rejected(ack: Callable[..., None], body: dict[str, Any], client: Any) -> None:
        note = _rejected(body, approvers=approvers, decisions=decisions, resume=resume, audit=audit)
        if note == COMMENT_REQUIRED:
            ack(response_action="errors", errors={"comment": "A comment is required."})
            return
        ack()
        if note:
            _ephemeral(body, client, note)

    app.event("app_mention")(mention)
    app.action("meridian_approve")(action)
    app.action("meridian_reject")(action)
    app.action("meridian_acknowledge")(action)
    app.view("meridian_reject_submit")(rejected)


def _worker(settings: Settings, saver: Any, ports: Any) -> Any:
    """Compile the graph. Resume only runs the gate, so the other chats are fakes."""
    evidence = Path("data/slack-evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    triage = FakeTriageModel()
    root = HeuristicRootCauseModel()
    return build_graph(
        num_cpus=settings.ray_num_cpus,
        model_name=triage.model_name,
        chat=triage,
        checkpointer=saver,
        root_model_name=root.model_name,
        root_chat=root,
        evidence_root=evidence,
        tools=[],
        sandbox_timeout_s=settings.sandbox_timeout_s,
        sandbox_memory_mb=settings.sandbox_memory_mb,
        max_row_loss=settings.max_row_loss,
        max_quarantine=settings.max_quarantine,
        max_outside_change=settings.max_outside_change,
        sample_seed=settings.llm_seed,
        wall_budget_s=settings.run_wall_budget_s,
        psi_min_rows=settings.psi_min_rows,
        approval=True,
        approval_ports=ports,
    )


def _action(
    body: dict[str, Any],
    *,
    approvers: frozenset[str],
    decisions: DecisionLog,
    resume: Callable[[str, dict[str, object]], None],
    audit: Any | None = None,
) -> str | dict[str, object]:
    from meridian.slack.handlers import on_action

    user = body.get("user", {})
    actions = body.get("actions", [])
    action = actions[0] if actions else {}
    channel = _channel(body)
    message = body.get("message", {})
    return on_action(
        user_id=str(user.get("id", "")),
        action_id=str(action.get("action_id", "")),
        value=str(action.get("value", "")),
        approvers=approvers,
        log=decisions,
        resume=resume,
        now=datetime.now(UTC),
        channel=channel,
        message_ts=str(message.get("ts", "")),
        audit=audit,
    )


def _rejected(
    body: dict[str, Any],
    *,
    approvers: frozenset[str],
    decisions: DecisionLog,
    resume: Callable[[str, dict[str, object]], None],
    audit: Any | None = None,
) -> str:
    view = body.get("view", {})
    state = view.get("state", {}).get("values", {})
    comment_block = state.get("comment", {}).get("comment", {})
    comment = comment_block.get("value") or ""
    user = body.get("user", {})
    return on_reject_submit(
        user_id=str(user.get("id", "")),
        comment=str(comment),
        metadata=str(view.get("private_metadata", "")),
        approvers=approvers,
        log=decisions,
        resume=resume,
        now=datetime.now(UTC),
        audit=audit,
    )


def _ephemeral_or_modal(body: dict[str, Any], client: Any, note: str | dict[str, object]) -> None:
    if isinstance(note, dict):
        client.views_open(trigger_id=body.get("trigger_id", ""), view=note)
        return
    if note:
        _ephemeral(body, client, note)


def _ephemeral(body: dict[str, Any], client: Any, text: str) -> None:
    user = body.get("user", {})
    client.chat_postEphemeral(channel=_channel(body), user=str(user.get("id", "")), text=text)


def _channel(body: dict[str, Any]) -> str:
    channel = body.get("channel", {})
    if isinstance(channel, dict) and channel.get("id"):
        return str(channel["id"])
    container = body.get("container", {})
    if isinstance(container, dict) and container.get("channel_id"):
        return str(container["channel_id"])
    return ""


def _secret(value: Any) -> str:
    if value is None:
        return ""
    reveal = getattr(value, "get_secret_value", None)
    if callable(reveal):
        return str(reveal())
    return str(value)


def _install(stop: Callable[[], None]) -> None:
    def _handle(signum: int, _frame: object) -> None:
        logger.info("signal=%s", signum)
        stop()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _ray() -> Any:
    import ray

    return ray


if __name__ == "__main__":
    raise SystemExit(main())
