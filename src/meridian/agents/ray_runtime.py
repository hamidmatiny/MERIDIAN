"""Ray runtime for Sentinel. A cluster this process already started is left running."""

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray

_LOCK = threading.Lock()
_GENERATION = 0


def prepare_ray_runtime() -> dict[str, dict[str, str]]:
    """Stop Ray from rebuilding the project with ``uv run`` inside each worker.

    The flag is read when ``ray`` is imported, so it is also forced off on the
    already-imported constant. The returned runtime env is the one passed to
    ``ray.init`` and does not upload a working directory.
    """
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    import ray._private.ray_constants as constants

    constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    return {"env_vars": {"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"}}


def start_ray(num_cpus: int) -> int:
    """Start one local cluster, or keep the one already running.

    The returned generation changes only when this process actually calls
    ``ray.init``. Callers use it to tell a new cluster from the one they warmed.
    """
    _claim(num_cpus)
    return _GENERATION


def _claim(num_cpus: int) -> bool:
    """Return True when this call started the cluster."""
    global _GENERATION
    if num_cpus < 1:
        raise ValueError("num_cpus must be >= 1")
    with _LOCK:
        if ray.is_initialized():
            return False
        os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
        store = os.environ.get("RAY_OBJECT_STORE_MEMORY", "").strip()
        try:
            if store:
                ray.init(
                    num_cpus=num_cpus,
                    include_dashboard=False,
                    logging_level=logging.ERROR,
                    ignore_reinit_error=True,
                    runtime_env=prepare_ray_runtime(),
                    object_store_memory=int(store),
                )
            else:
                ray.init(
                    num_cpus=num_cpus,
                    include_dashboard=False,
                    logging_level=logging.ERROR,
                    ignore_reinit_error=True,
                    runtime_env=prepare_ray_runtime(),
                )
        except AssertionError as exc:
            # A raced second init finds the driver already connected. Use that cluster.
            if "ray.init twice" not in str(exc) or not ray.is_initialized():
                raise
            return False
        _GENERATION += 1
        return True


@contextmanager
def shared_ray(num_cpus: int) -> Iterator[None]:
    """Hold one cluster for a whole run so Sentinel and the sandbox share it."""
    started = _claim(num_cpus)
    try:
        yield
    finally:
        if started:
            ray.shutdown()


@contextmanager
def ray_runtime(num_cpus: int) -> Iterator[None]:
    """Initialize Ray for this block and shut it down afterwards.

    If this process already has a cluster, the block uses it and does not shut it down.
    """
    started = _claim(num_cpus)
    try:
        yield
    finally:
        if started:
            ray.shutdown()
