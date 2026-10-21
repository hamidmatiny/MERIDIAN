"""Read token usage from a model response.

Anthropic sometimes leaves ``usage_metadata`` at 0/0 and puts the real counts on
``response_metadata["usage"]``. A paid response with no usage at all is not a
zero-token call: the caller aborts rather than record a guess.
"""

from typing import Any

from meridian.contracts.llm_call import LLMCall


class TimeoutNote:
    """One timed-out hop. The node records it as an LLMCall."""

    def __init__(self, provider: str, model: str, latency_ms: int) -> None:
        self.provider = provider
        self.model = model
        self.latency_ms = latency_ms


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
    structured: bool = False,
    corrective: bool = False,
) -> LLMCall:
    """Build one call record from the response and the bound client."""
    usage = read_usage(raw)
    incoming, outgoing = (0, 0) if usage is None else usage
    owner = _owner(bound)
    provider = _text(getattr(owner, "provider_name", None), "fake")
    active = getattr(owner, "active_model", None)
    chosen = active if isinstance(active, str) and active.strip() else model
    measured = getattr(owner, "last_latency_ms", 0)
    latency = latency_ms if latency_ms > 0 else _whole(measured)
    loaded = _load_ms(raw)
    if loaded == 0:
        loaded = _whole(getattr(owner, "last_load_ms", 0))
    return LLMCall(
        node=node,
        model=chosen,
        provider=provider,
        prompt_version=prompt_version,
        input_tokens=incoming,
        output_tokens=outgoing,
        prompt_tokens_before=tokens_before,
        prompt_tokens_after=tokens_after,
        latency_ms=max(latency, 0),
        attempt=attempt,
        succeeded=succeeded,
        fallback_used=bool(getattr(owner, "fallback_used", False)),
        structured=structured,
        temperature=_float(getattr(owner, "temperature", 0.0)),
        seed=_seed(getattr(owner, "seed", 42)),
        corrective=corrective,
        load_duration_ms=loaded,
    )


def assemble_calls(
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
    structured: bool = False,
    corrective: bool = False,
) -> list[LLMCall]:
    """The timed-out hops that preceded this attempt, then the attempt itself."""
    owner = _owner(bound)
    notes = list(getattr(owner, "timeouts", []))
    if hasattr(owner, "timeouts"):
        owner.timeouts.clear()
    number = attempt
    prior: list[LLMCall] = []
    for note in notes:
        if not isinstance(note, TimeoutNote):
            continue
        prior.append(
            LLMCall(
                node=node,
                model=note.model,
                provider=note.provider,
                prompt_version=prompt_version,
                input_tokens=0,
                output_tokens=0,
                latency_ms=max(note.latency_ms, 0),
                attempt=number,
                succeeded=False,
                structured=structured,
                temperature=_float(getattr(owner, "temperature", 0.0)),
                seed=_seed(getattr(owner, "seed", 42)),
                timed_out=True,
            )
        )
        number += 1
    primary = make_llm_call(
        node=node,
        model=model,
        prompt_version=prompt_version,
        raw=raw,
        latency_ms=latency_ms,
        attempt=number,
        succeeded=succeeded,
        bound=bound,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        structured=structured,
        corrective=corrective,
    )
    return [*prior, primary]


def _owner(bound: object) -> object:
    parent = getattr(bound, "parent", None)
    return parent if parent is not None else bound


def _load_ms(raw: object) -> int:
    meta = getattr(raw, "response_metadata", None)
    if isinstance(raw, dict):
        nested = raw.get("response_metadata")
        meta = nested if isinstance(nested, dict) else raw
    if not isinstance(meta, dict):
        return 0
    value = meta.get("load_duration")
    return _whole(value) // 1_000_000 if _whole(value) else 0


def _whole(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 42
    return value


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
