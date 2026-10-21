"""Provider failure classes. Billing, auth, and quota stop the run."""

from enum import StrEnum
from typing import Any

_FATAL = frozenset({"BILLING", "AUTH", "QUOTA"})


class ProviderFailure(StrEnum):
    """How a provider error is allowed to affect the rest of the run."""

    BILLING = "BILLING"
    AUTH = "AUTH"
    QUOTA = "QUOTA"
    TRANSIENT = "TRANSIENT"
    UNSUPPORTED = "UNSUPPORTED"
    BAD_OUTPUT = "BAD_OUTPUT"


class RunAborted(Exception):  # noqa: N818
    """A billing, auth, or quota failure. Nodes must not continue after this."""

    def __init__(self, kind: str, detail: str) -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")


class AbortGate:
    """Process-local stop shared by every node in one run."""

    def __init__(self) -> None:
        self.reason: str | None = None

    def trip(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason

    def check(self) -> None:
        if self.reason is not None:
            raise RunAborted("ABORTED", self.reason)


def unwrap_abort(exc: BaseException) -> RunAborted | None:
    """Find a ``RunAborted`` wrapped by the graph runtime."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, RunAborted):
            return current
        seen.add(id(current))
        cause = current.__cause__
        context = current.__context__
        current = cause if isinstance(cause, BaseException) else context
    return None


def classify(exc: BaseException) -> ProviderFailure:
    """Map a provider exception onto the circuit-breaker classes."""
    status = _status(exc)
    text = f"{type(exc).__name__} {exc}".lower()
    if status == 402 or (status == 400 and "credit" in text):
        return ProviderFailure.BILLING
    if status in (401, 403):
        return ProviderFailure.AUTH
    if status == 429 and ("quota" in text or "limit" in text):
        return ProviderFailure.QUOTA
    if status == 501 or _capability_missing(text):
        return ProviderFailure.UNSUPPORTED
    if status == 429 or (status is not None and status >= 500):
        return ProviderFailure.TRANSIENT
    if "timeout" in text or "timed out" in text:
        return ProviderFailure.TRANSIENT
    return ProviderFailure.BAD_OUTPUT


def call_model(bound: Any, messages: Any) -> tuple[object | None, int, int]:
    """Invoke one turn. Fatal errors propagate. Other errors become a failed turn."""
    from meridian.llm.prompt_size import prepare_messages

    prepared, before, after = prepare_messages(messages)
    try:
        return guarded_invoke(bound, prepared), before, after
    except RunAborted:
        raise
    except Exception:
        return None, before, after


def guarded_invoke(bound: Any, messages: Any) -> Any:
    """Call ``bound``. A fatal provider error becomes ``RunAborted``."""
    gate = getattr(bound, "abort_gate", None)
    if isinstance(gate, AbortGate):
        gate.check()
    try:
        return bound.invoke(messages)
    except RunAborted as exc:
        if isinstance(gate, AbortGate):
            gate.trip(str(exc))
        raise
    except Exception as exc:
        kind = classify(exc)
        if kind.value in _FATAL:
            if isinstance(gate, AbortGate):
                gate.trip(f"{kind.value}: {exc}")
            raise RunAborted(kind.value, str(exc)) from exc
        raise


def _capability_missing(text: str) -> bool:
    """True when the provider says a feature is not implemented."""
    if "unavailable" in text and ("structured" in text or "capability" in text):
        return True
    return "does not support" in text and ("tool" in text or "json" in text or "structured" in text)


def _status(exc: BaseException) -> int | None:
    for name in ("status_code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
