"""One-command demo. Fake models, a terminal approval, an applied patch.

``fake models: demonstrates the pipeline, not model quality``
"""

import os
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from meridian.agents.ray_runtime import shared_ray
from meridian.audit.store import install_from_env, note
from meridian.contracts.fault_type import FaultType
from meridian.contracts.state import MeridianState
from meridian.graph.build import build_graph
from meridian.mcp_servers.client import inprocess_tools
from meridian.run import PreparedChats, _chat, _prepare
from meridian.slack.gate import ApprovalPorts
from meridian.slack.gateway import RecordingGateway
from meridian.slack.store import MessageLog, OpenIndex
from meridian.tracing import enabled as tracing_enabled
from meridian.tracing import run_span

_BANNER = "fake models: demonstrates the pipeline, not model quality"
_FAULTS = (FaultType.UNIT_MISMATCH, FaultType.NULL_EXPLOSION)


def main(argv: list[str] | None = None) -> int:
    """Inject two faults, run the graph, and ask approve or reject in the terminal."""
    del argv
    started = time.perf_counter()
    print(_BANNER, flush=True)
    from meridian.config import get_settings

    settings = get_settings()
    print(install_from_env(Path("data/audit")), flush=True)
    tracing_enabled()
    root = Path("data/demo")
    root.mkdir(parents=True, exist_ok=True)
    chats = _chat(settings, live=False)
    with shared_ray(1):
        for index, fault in enumerate(_FAULTS):
            _run_fault(root / fault.value, fault, seed=20 + index, chats=chats, rows=800)
    elapsed = time.perf_counter() - started
    print(f"wall_s={elapsed:.3f}", flush=True)
    return 0


def _run_fault(
    folder: Path, fault: FaultType, *, seed: int, chats: PreparedChats, rows: int
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    prepared = _prepare(
        folder,
        dataset="vehicle_telemetry",
        fault=fault,
        intensity=0.5,
        seed=seed,
        rows=rows,
        partition=date(2026, 10, 24),
    )
    ports = ApprovalPorts(
        gateway=RecordingGateway(MessageLog(folder / "messages")),
        apply_root=folder / "applied",
        postmortem_root=folder / "postmortems",
        channel="console",
        open_index=OpenIndex(folder / "open.json"),
    )
    run_id = uuid4()
    config: dict[str, Any] = {
        "configurable": {"thread_id": str(run_id)},
        "recursion_limit": 40,
    }
    print(f"fault={fault.value} run_id={run_id}", flush=True)
    with SqliteSaver.from_conn_string(str(folder / "ckpt.sqlite")) as saver:
        graph = build_graph(
            num_cpus=1,
            model_name=chats.triage_model,
            chat=chats.triage_chat,
            checkpointer=saver,
            root_model_name=chats.root_model,
            root_chat=chats.root_chat,
            evidence_root=folder,
            tools=inprocess_tools(folder),
            fix_model_name=chats.fix_model,
            fix_chat=chats.fix_chat,
            critic_model_name=chats.critic_model,
            critic_chat=chats.critic_chat,
            approval=True,
            approval_ports=ports,
            psi_min_rows=200,
        )
        with run_span(str(run_id)):
            try:
                result = graph.invoke(  # type: ignore[call-overload]
                    MeridianState(run_id=run_id, partitions=[prepared.ref]),
                    config,
                )
            except Exception as exc:
                if type(exc).__name__ != "GraphInterrupt":
                    raise
                result = None
            if result is None or _paused(result):
                _resume(graph, config, _ask())
            note(
                "runs",
                {"run_id": str(run_id), "dataset": "vehicle_telemetry", "fault": fault.value},
            )
    print(f"applied={_names(folder / 'applied')}", flush=True)
    print(f"postmortem={_names(folder / 'postmortems')}", flush=True)


def _paused(result: object) -> bool:
    return isinstance(result, dict) and bool(result.get("__interrupt__"))


def _resume(graph: Any, config: dict[str, Any], choice: str) -> None:
    decision = "APPROVED" if choice == "approve" else "REJECTED"
    payload = {
        "decision": decision,
        "approver": "console",
        "decided_at": datetime.now(UTC).isoformat(),
        "comment": "",
    }
    graph.invoke(Command(resume=payload), config)


def _ask() -> str:
    forced = os.environ.get("MERIDIAN_DEMO_DECISION", "").strip().lower()
    print("approval prompt: type approve or reject", flush=True)
    if forced in {"approve", "reject"}:
        print(forced, flush=True)
        return forced
    if not sys.stdin.isatty():
        print("approve", flush=True)
        return "approve"
    while True:
        answer = input("> ").strip().lower()
        if answer in {"approve", "reject"}:
            return answer
        print("type approve or reject")


def _names(folder: Path) -> str:
    if not folder.is_dir():
        return "(none)"
    found = sorted(path.name for path in folder.iterdir())
    return ", ".join(found) if found else "(none)"


if __name__ == "__main__":
    raise SystemExit(main())
