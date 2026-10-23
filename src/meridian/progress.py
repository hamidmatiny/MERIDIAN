"""One stderr line per node, and a heartbeat while a model call is in flight.

stdout stays the final state JSON. These records go through the ``meridian.progress``
logger, which the CLI and slack_app both attach to stderr. The same boundary is appended
to ``data/progress/<run_id>.jsonl`` so a postmortem can use the real timestamps.
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from meridian.contracts.progress_event import ProgressEvent

logger = logging.getLogger("meridian.progress")

HEARTBEAT_S = 30.0


def progress_dir() -> Path:
    """Directory of per-run JSONL files. Tests point this at a temp dir."""
    return Path(os.environ.get("MERIDIAN_PROGRESS_DIR", "data/progress"))


def progress_path(run_id: UUID) -> Path:
    """The append-only file for one run."""
    return progress_dir() / f"{run_id}.jsonl"


def node_line(
    run_id: UUID,
    case: str,
    node: str,
    attempt: int,
    elapsed_s: float,
    phase: str,
) -> None:
    """``phase`` is ``start`` or ``end``. The wall time is recorded here, not reconstructed."""
    logger.info(
        "run_id=%s case=%s node=%s attempt=%s elapsed_s=%.1f phase=%s",
        run_id,
        case,
        node,
        attempt,
        elapsed_s,
        phase,
    )
    recorded: Literal["start", "end"]
    if phase == "start":
        recorded = "start"
    elif phase == "end":
        recorded = "end"
    else:
        return
    event = ProgressEvent(
        occurred_at=datetime.now(UTC),
        run_id=run_id,
        case_id=case or "-",
        node=node,
        attempt=attempt,
        phase=recorded,
        elapsed_s=elapsed_s,
    )
    path = progress_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(event.model_dump_json() + "\n")


def load_progress(run_id: UUID) -> list[ProgressEvent]:
    """Events written for this run, in append order. Missing file means none were recorded."""
    path = progress_path(run_id)
    if not path.is_file():
        return []
    found: list[ProgressEvent] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            found.append(ProgressEvent.model_validate_json(raw))
    return found


def heartbeat(node: str, model: str, elapsed_s: float) -> None:
    """Logged on the heartbeat interval for the whole time a call is in flight."""
    logger.info("heartbeat node=%s model=%s elapsed_s=%.1f", node, model, elapsed_s)
