"""Read token usage from a model response.

Anthropic sometimes leaves ``usage_metadata`` at 0/0 and puts the real counts on
``response_metadata["usage"]``. A paid response with no usage at all is not a
zero-token call: the caller aborts rather than record a guess.
"""

from typing import Any

from meridian.contracts.llm_call import LLMCall

_USAGE_KEYS = ("usage", "token_usage", "usage_metadata")


def read_usage(message: object) -> tuple[int, int] | None:
    """Return ``(input, output)``, or None when the response carries no usage."""
    found: list[tuple[int, int]] = []
    if isinstance(message, dict):
        _collect(found, message.get("usage_metadata"))
        _collect(found, message.get("usage"))
        _collect_response(found, message.get("response_metadata"))
        if "raw" in message:
            nested = read_usage(message.get("raw"))
            if nested is not None:
                found.append(nested)
    else:
        _collect(found, getattr(message, "usage_metadata", None))
        _collect_response(found, getattr(message, "response_metadata", None))
    if not found:
        return None
    for item in found:
        if item != (0, 0):
            return item
    return found[0]


def make_llm_call(
    *,
    node: str,
    model: str,
    prompt_version: str,
    raw: object,
    latency_ms: int,
    attempt: int,
    succeeded: bool,
    bound: object,
    tokens_before: int,
    tokens_after: int,
) -> LLMCall:
    """Build one call record from the response and the bound client."""
    usage = read_usage(raw)
    incoming, outgoing = (0, 0) if usage is None else usage
    provider = _text(getattr(bound, "provider_name", None), "fake")
    active = getattr(bound, "active_model", None)
    chosen = active if isinstance(active, str) and active.strip() else model
    return LLMCall(
        node=node,
        model=chosen,
        provider=provider,
        prompt_version=prompt_version,
        input_tokens=incoming,
        output_tokens=outgoing,
        prompt_tokens_before=tokens_before,
        prompt_tokens_after=tokens_after,
        latency_ms=max(latency_ms, 0),
        attempt=attempt,
        succeeded=succeeded,
        fallback_used=bool(getattr(bound, "fallback_used", False)),
    )


def _collect(found: list[tuple[int, int]], value: object) -> None:
    pair = _pair(value)
    if pair is not None:
        found.append(pair)


def _collect_response(found: list[tuple[int, int]], value: object) -> None:
    if not isinstance(value, dict):
        return
    for key in _USAGE_KEYS:
        _collect(found, value.get(key))
    _collect(found, value)


def _pair(value: object) -> tuple[int, int] | None:
    if isinstance(value, dict):
        incoming = _token(
            value.get("input_tokens", value.get("prompt_tokens")),
        )
        outgoing = _token(
            value.get("output_tokens", value.get("completion_tokens")),
        )
    else:
        incoming = _token(getattr(value, "input_tokens", None))
        outgoing = _token(getattr(value, "output_tokens", None))
    if incoming is None or outgoing is None:
        return None
    return incoming, outgoing


def _token(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return max(value, 0)


def _text(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return default
