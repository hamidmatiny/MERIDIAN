"""OpenTelemetry spans. A no-op until an exporter is configured.

Spans never carry a prompt body or a secret. The default path does not build a
provider, so tests that leave the exporter unset do not export anything.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_FORBIDDEN = frozenset(
    {
        "prompt",
        "prompt_text",
        "prompt_body",
        "messages",
        "message",
        "secret",
        "password",
        "authorization",
        "api_key",
        "token",
        "app_token",
        "bot_token",
    }
)
_Attr = str | int | float | bool
_current: ContextVar["Span | None"] = ContextVar("meridian_span", default=None)
_stack: list["Span"] = []
_enabled = False
_configured = False
_cases: dict[str, "Span"] = {}


class Span:
    """One span. Attribute writes of secrets and prompt bodies are ignored."""

    def __init__(self, name: str, otel: Any) -> None:
        self.name = name
        self.attributes: dict[str, _Attr] = {}
        self._otel = otel
        self._ended = False

    def set_attribute(self, key: str, value: _Attr) -> None:
        if not _allowed(key):
            return
        self.attributes[key] = value
        if self._otel is not None:
            self._otel.set_attribute(key, value)

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        if self._otel is not None:
            self._otel.end()


class _Null(Span):
    def __init__(self) -> None:
        super().__init__("", None)

    def set_attribute(self, key: str, value: _Attr) -> None:
        del key, value

    def end(self) -> None:
        return


_NULL = _Null()


def enabled() -> bool:
    """True after an exporter has been installed."""
    _configure_once()
    return _enabled


def reset() -> None:
    """Return to the no-op tracer. A test that installed an exporter calls this after."""
    global _enabled, _configured
    _enabled = False
    _configured = True
    _cases.clear()
    _stack.clear()


def install_memory(exporter: Any) -> None:
    """Send spans to an in-memory exporter. Tests use this."""
    _install(exporter, console=False, endpoint="")


@contextmanager
def span(name: str, *, parent: Span | None = None) -> Iterator[Span]:
    """Open a span and make it the parent of spans opened inside the block."""
    chosen = parent if parent is not None else (_stack[-1] if _stack else _current.get())
    opened = start_span(name, chosen)
    _stack.append(opened)
    token = _current.set(opened)
    try:
        yield opened
    finally:
        _current.reset(token)
        if _stack and _stack[-1] is opened:
            _stack.pop()
        else:
            _stack[:] = [item for item in _stack if item is not opened]
        opened.end()


def start_span(name: str, parent: Span | None = None) -> Span:
    """Open a span that the caller ends. A disabled tracer returns a no-op."""
    if not enabled():
        return _NULL
    otel = _otel_span(name, parent)
    return Span(name, otel)


@contextmanager
def run_span(run_id: str) -> Iterator[Span]:
    """The root span for one graph run. Leftover case spans close with it."""
    current = _current.get()
    if current is not None and current.name == "meridian.run":
        yield current
        return
    _cases.clear()
    with span("meridian.run") as opened:
        opened.set_attribute("run_id", run_id)
        try:
            yield opened
        finally:
            for item in list(_cases.values()):
                item.end()
            _cases.clear()


def case_parent(run_id: str, case_id: str) -> Span:
    """The open case span, or the current span when this node has no case."""
    if not enabled() or case_id == "-":
        current = _current.get()
        return current if current is not None else _NULL
    key = f"{run_id}:{case_id}"
    found = _cases.get(key)
    if found is not None:
        return found
    opened = start_span("meridian.case", _current.get())
    opened.set_attribute("case_id", case_id)
    _cases[key] = opened
    return opened


def finish_case(run_id: str, case_id: str, result: dict[str, Any]) -> None:
    """Record the case outcome and close its span."""
    opened = _cases.pop(f"{run_id}:{case_id}", None)
    if opened is None:
        return
    opened.set_attribute("outcome", _category(result))
    opened.end()


def note_llm(
    *,
    provider: str,
    model: str,
    prompt_version: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    timed_out: bool,
    think: bool,
    attempt: int,
) -> None:
    """One model attempt. The prompt text is not an attribute."""
    if not enabled():
        return
    with span("meridian.llm") as opened:
        opened.set_attribute("provider", provider)
        opened.set_attribute("model", model)
        opened.set_attribute("prompt_version", prompt_version)
        opened.set_attribute("input_tokens", input_tokens)
        opened.set_attribute("output_tokens", output_tokens)
        opened.set_attribute("latency_ms", latency_ms)
        opened.set_attribute("timed_out", timed_out)
        opened.set_attribute("think", think)
        opened.set_attribute("attempt", attempt)


def note_tool(name: str, ref_count: int) -> None:
    """One tool call. ``ref_count`` is how many evidence refs the result carried."""
    if not enabled():
        return
    with span("meridian.tool") as opened:
        opened.set_attribute("tool", name)
        opened.set_attribute("ref_count", ref_count)


def note_sandbox(entry: str, kind: str, rows_in: int, rows_out: int) -> None:
    """One sandboxed remediation. The source text is not an attribute."""
    if not enabled():
        return
    with span("meridian.sandbox") as opened:
        opened.set_attribute("entry", entry)
        opened.set_attribute("kind", kind)
        opened.set_attribute("rows_in", rows_in)
        opened.set_attribute("rows_out", rows_out)


def _configure_once() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    console = os.environ.get("MERIDIAN_TRACE", "").strip().lower() == "console"
    if not endpoint and not console:
        return
    _install(None, console=console, endpoint=endpoint)


def _install(exporter: Any, *, console: bool, endpoint: str) -> None:
    global _enabled, _configured
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "meridian"}))
    once = getattr(trace, "_TRACER_PROVIDER_SET_ONCE", None)
    if once is not None:
        once._done = False
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    if console:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    if endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            SimpleSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=not endpoint.startswith("https://"))
            )
        )
    trace.set_tracer_provider(provider)
    _enabled = True
    _configured = True


def _otel_span(name: str, parent: Span | None) -> Any:
    from opentelemetry import trace

    kwargs: dict[str, Any] = {}
    otel_parent = None if parent is None else parent._otel
    if otel_parent is not None:
        kwargs["context"] = trace.set_span_in_context(otel_parent)
    return trace.get_tracer("meridian").start_span(name, **kwargs)


def _allowed(key: str) -> bool:
    lowered = key.lower()
    if lowered in _FORBIDDEN:
        return False
    if "secret" in lowered or "password" in lowered or "authorization" in lowered:
        return False
    return not (lowered == "token" or lowered.endswith("_token"))


def _category(result: dict[str, Any]) -> str:
    cases = result.get("cases")
    if not isinstance(cases, dict) or not cases:
        return "OTHER"
    case = next(iter(cases.values()))
    verification = getattr(case, "verification", None)
    if verification is not None and bool(getattr(verification, "passed", False)):
        return "VERIFIED_AWAITING_APPROVAL"
    if bool(getattr(case, "needs_human", False)):
        return "AGENT_ESCALATED"
    if getattr(case, "patch", None) is None:
        return "NO_PATCH"
    if int(getattr(case, "critic_rounds", 0)) >= 3:
        return "CRITIC_LIMIT"
    if int(getattr(case, "fix_attempts", 0)) >= 3:
        return "FIX_ATTEMPT_LIMIT"
    return "OTHER"
