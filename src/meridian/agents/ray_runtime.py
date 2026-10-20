"""Ray runtime for Sentinel. A cluster this process already started is left running."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray


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


@contextmanager
def shared_ray(num_cpus: int) -> Iterator[None]:
    """Hold one cluster for a whole run so Sentinel and the sandbox share it."""
    if num_cpus < 1:
        raise ValueError("num_cpus must be >= 1")
    started = False
    if not ray.is_initialized():
        os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
        ray.init(
            num_cpus=num_cpus,
            include_dashboard=False,
            logging_level=logging.ERROR,
            ignore_reinit_error=True,
            runtime_env=prepare_ray_runtime(),
        )
        started = True
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
    if num_cpus < 1:
        raise ValueError("num_cpus must be >= 1")
    if ray.is_initialized():
        yield
        return
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    ray.init(
        num_cpus=num_cpus,
        include_dashboard=False,
        logging_level=logging.ERROR,
        ignore_reinit_error=False,
        runtime_env=prepare_ray_runtime(),
    )
    try:
        yield
    finally:
        ray.shutdown()
