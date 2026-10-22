"""One stderr line per node, and a heartbeat while a model call is in flight.

stdout stays the final state JSON. These records go through the ``meridian.progress``
logger, which the CLI and slack_app both attach to stderr.
"""

import logging
from uuid import UUID

logger = logging.getLogger("meridian.progress")

HEARTBEAT_S = 30.0


def node_line(
    run_id: UUID,
    case: str,
    node: str,
    attempt: int,
    elapsed_s: float,
    phase: str,
) -> None:
    """``phase`` is ``start`` or ``end``."""
    logger.info(
        "run_id=%s case=%s node=%s attempt=%s elapsed_s=%.1f phase=%s",
        run_id,
        case,
        node,
        attempt,
        elapsed_s,
        phase,
    )


def heartbeat(node: str, model: str, elapsed_s: float) -> None:
    """Logged on the heartbeat interval for the whole time a call is in flight."""
    logger.info("heartbeat node=%s model=%s elapsed_s=%.1f", node, model, elapsed_s)
