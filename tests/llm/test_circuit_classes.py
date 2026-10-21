"""Circuit-breaker classes, including a wrapped abort and a missing usage object."""

import pytest
from langchain_core.messages import HumanMessage

from meridian.llm.errors import (
    AbortGate,
    ProviderFailure,
    RunAborted,
    call_model,
    classify,
    guarded_invoke,
    unwrap_abort,
)
from meridian.llm.usage import make_llm_call, read_usage


def test_status_classes() -> None:
    billing = RuntimeError("credit balance is too low")
    billing.status_code = 400
    assert classify(billing) is ProviderFailure.BILLING
    forbidden = RuntimeError("forbidden")
    forbidden.status_code = 403
    assert classify(forbidden) is ProviderFailure.AUTH
    quota = RuntimeError("quota exceeded")
    quota.status_code = 429
    assert classify(quota) is ProviderFailure.QUOTA
    busy = RuntimeError("again")
    busy.status_code = 429
    assert classify(busy) is ProviderFailure.TRANSIENT
    down = RuntimeError("unavailable")
    down.status_code = 503
    assert classify(down) is ProviderFailure.TRANSIENT
    wrapped = RuntimeError("upstream")
    response = RuntimeError("body")
    response.status_code = 500
    wrapped.response = response
    assert classify(wrapped) is ProviderFailure.TRANSIENT
    assert classify(TimeoutError("timed out")) is ProviderFailure.TRANSIENT
    missing = RuntimeError("structured output is unavailable")
    missing.status_code = 501
    assert classify(missing) is ProviderFailure.UNSUPPORTED
    assert classify(RuntimeError("model does not support tools")) is ProviderFailure.UNSUPPORTED
    assert classify(ValueError("schema")) is ProviderFailure.BAD_OUTPUT


def test_unwrap_walks_a_cycle_and_a_cause() -> None:
    left = RuntimeError("left")
    right = RuntimeError("right")
    left.__cause__ = right
    right.__cause__ = left
    assert unwrap_abort(left) is None
    inner = RunAborted("BILLING", "credit")
    outer = RuntimeError("wrapped")
    outer.__cause__ = inner
    assert unwrap_abort(outer) is inner


def test_guarded_invoke_stops_on_a_tripped_gate() -> None:
    gate = AbortGate()
    gate.trip("BILLING: credit")
    gate.trip("ignored")

    class _Bound:
        abort_gate = gate

        def invoke(self, messages: object) -> object:
            del messages
            return "no"

    with pytest.raises(RunAborted, match="ABORTED"):
        guarded_invoke(_Bound(), [])


def test_guarded_invoke_records_a_fatal_error() -> None:
    gate = AbortGate()

    class _Bound:
        abort_gate = gate

        def invoke(self, messages: object) -> object:
            del messages
            error = RuntimeError("credit balance")
            error.status_code = 402
            raise error

    with pytest.raises(RunAborted, match="BILLING"):
        guarded_invoke(_Bound(), [])
    assert gate.reason is not None and gate.reason.startswith("BILLING")


def test_guarded_invoke_keeps_an_already_aborted_call() -> None:
    gate = AbortGate()

    class _Bound:
        abort_gate = gate

        def invoke(self, messages: object) -> object:
            del messages
            raise RunAborted("QUOTA", "limit")

    with pytest.raises(RunAborted, match="QUOTA"):
        guarded_invoke(_Bound(), [])
    assert gate.reason == "QUOTA: limit"


def test_call_model_returns_none_for_a_bad_output() -> None:
    class _Bound:
        def invoke(self, messages: object) -> object:
            del messages
            raise ValueError("not json")

    result, before, after = call_model(_Bound(), [HumanMessage(content="hello")])
    assert result is None
    assert before == after


def test_usage_nested_raw_and_blank_provider() -> None:
    found = read_usage({"raw": {"usage": {"input_tokens": 4, "output_tokens": 1}}})
    assert found == (4, 1)
    assert read_usage({"usage_metadata": {"input_tokens": True, "output_tokens": 1}}) is None
    call = make_llm_call(
        node="triage",
        model="qwen3.5:4b",
        prompt_version="triage.v1",
        raw=None,
        latency_ms=-1,
        attempt=1,
        succeeded=True,
        bound=type("Bound", (), {"provider_name": "  ", "active_model": "  "})(),
        tokens_before=3,
        tokens_after=2,
    )
    assert call.provider == "fake"
    assert call.model == "qwen3.5:4b"
    assert call.latency_ms == 0
