"""Ray runtime for Sentinel. One process-wide init, shutdown on exit."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

import ray


@contextmanager
def ray_runtime(num_cpus: int) -> Iterator[None]:
    """Initialize Ray for this block and call ``ray.shutdown`` on the way out."""
    if num_cpus < 1:
        raise ValueError("num_cpus must be >= 1")
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
