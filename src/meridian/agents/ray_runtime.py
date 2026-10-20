"""Ray runtime for Sentinel. A cluster this process already started is left running."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

import ray


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
    )
    try:
        yield
    finally:
        ray.shutdown()
