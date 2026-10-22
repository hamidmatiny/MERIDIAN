"""Process shutdown shared by the signal handlers."""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("meridian.slack")


def shutdown_process(
    *,
    close_socket: Callable[[], None],
    sessions: list[Any],
    ray_module: Any,
) -> None:
    """Close the socket, any MCP session, and Ray."""
    logger.info("shutting down")
    close_socket()
    for session in sessions:
        closer = getattr(session, "close", None)
        if callable(closer):
            closer()
    if ray_module.is_initialized():
        ray_module.shutdown()
