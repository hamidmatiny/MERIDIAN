"""Process-start wall budget.

The clock starts when the CLI starts, before probes, model load, Ray, MCP, and
the graph. A SIGALRM watchdog interrupts the main thread. Nodes that block on
another thread are abandoned by ``call_bounded``, which polls the same clock.
"""

import signal
import threading
from collections.abc import Callable
from types import FrameType
from typing import Any, TypeVar

from meridian.llm.errors import RunAborted

T = TypeVar("T")
_Handler = Callable[[int, FrameType | None], Any] | int | None


class WallClock:
    """One run's budget, measured from ``start`` with ``time.monotonic``."""

    def __init__(self, budget_s: float) -> None:
        import time

        self.budget_s = budget_s
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._watching = False
        self._tripped = False
        self._previous: _Handler = None

    def elapsed(self) -> float:
        import time

        return time.monotonic() - self.started

    def remaining(self) -> float:
        return max(self.budget_s - self.elapsed(), 0.0)

    def expired(self) -> bool:
        return self.elapsed() >= self.budget_s

    def check(self) -> None:
        if self.expired():
            self.trip()
            raise RunAborted("WALL_BUDGET", "run wall budget exceeded")

    def trip(self) -> None:
        """Remember that this budget already aborted, and cancel a pending alarm."""
        self._tripped = True
        self.disarm()

    def start(self) -> None:
        """Arm SIGALRM for the main thread. Node threads poll ``expired``."""
        if self._watching:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("the wall watchdog starts on the main thread")
        self._previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._on_alarm)
        self._watching = True
        self.arm()

    def arm(self) -> None:
        """Resume the main-thread alarm for whatever budget remains."""
        if not self._watching or self._stop.is_set() or self.expired():
            return
        signal.setitimer(signal.ITIMER_REAL, max(self.remaining(), 1e-6))

    def disarm(self) -> None:
        """Drop the alarm while a node poll is enforcing the same deadline."""
        if not self._watching:
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)

    def stop(self) -> None:
        self._stop.set()
        if self._watching:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        self._restore()

    def _restore(self) -> None:
        if not self._watching:
            return
        handler: _Handler = signal.SIG_DFL if self._previous is None else self._previous
        signal.signal(signal.SIGALRM, handler)
        self._previous = None
        self._watching = False

    def _on_alarm(self, signum: int, frame: FrameType | None) -> None:
        del signum, frame
        if self._tripped:
            return
        self.trip()
        raise RunAborted("WALL_BUDGET", "run wall budget exceeded")


def call_bounded(action: Callable[[], T], clock: WallClock, *, poll_s: float = 0.2) -> T:
    """Run ``action`` on a daemon thread and abandon it when the clock expires."""
    clock.check()
    done = threading.Event()
    outcome: list[T] = []
    problem: list[BaseException] = []

    def _run() -> None:
        try:
            outcome.append(action())
        except BaseException as exc:
            problem.append(exc)
        finally:
            done.set()

    threading.Thread(target=_run, name="meridian-bounded", daemon=True).start()
    while not done.is_set():
        left = clock.remaining()
        if left <= 0:
            clock.trip()
            raise RunAborted("WALL_BUDGET", "run wall budget exceeded")
        done.wait(min(poll_s, left))
    if problem:
        raise problem[0]
    return outcome[0]
