"""Free-tier routing: explicit model ids, family split, and no paid default."""

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from meridian.config import MeridianConfigError, Settings
from meridian.llm.budget import Budget, ModelPrice, SpendStore
from meridian.llm.errors import AbortGate, RunAborted
from meridian.llm.family import model_family
from meridian.llm.router import FallbackChat, Hop, explicit_model, resolve_chain
from meridian.llm.usage import read_usage


def test_omniroute_rejects_auto_and_combos() -> None:
    with pytest.raises(MeridianConfigError, match="explicit model id"):
        explicit_model("auto")
    with pytest.raises(MeridianConfigError, match="benchmark"):
        explicit_model("qwen3.5+gemma4")
    with pytest.raises(MeridianConfigError, match="combo"):
        explicit_model("combo-router")
    assert explicit_model("qwen3.5:4b") == "qwen3.5:4b"


def test_default_chain_excludes_paid_providers() -> None:
    hops = resolve_chain("ollama", "qwen3.5:4b", "ollama,omniroute", allow_paid=False)
    assert hops == [("ollama", "qwen3.5:4b"), ("omniroute", "qwen3.5:4b")]
    paid = resolve_chain(
        "ollama",
        "qwen3.5:4b",
        "ollama,omniroute,anthropic",
        allow_paid=False,
    )
    assert [name for name, _model in paid] == ["ollama", "omniroute"]


def test_paid_primary_requires_the_allow_flag() -> None:
    with pytest.raises(MeridianConfigError, match="MERIDIAN_ALLOW_PAID"):
        resolve_chain("anthropic", "claude-haiku", "ollama", allow_paid=False)


def test_qwen_fix_with_qwen_critic_is_rejected() -> None:
    with pytest.raises(ValidationError, match="different family"):
        Settings(
            _env_file=None,
            fix_model="qwen3.5:4b",
            critic_model="qwen3.5:9b",
        )


def test_qwen_fix_with_gemma_critic_is_allowed() -> None:
    settings = Settings(
        _env_file=None,
        fix_model="qwen3.5:4b",
        critic_model="gemma4:e4b",
    )
    assert model_family(settings.fix_model or "") == "qwen"
    assert model_family(settings.critic_model or "") == "gemma"


def test_allow_paid_zero_is_false() -> None:
    settings = Settings(_env_file=None, allow_paid="0")
    assert settings.allow_paid is False


def test_transient_failure_falls_back_after_two_retries() -> None:
    class _Down:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: object, **kwargs: object) -> object:
            del messages, kwargs
            self.calls += 1
            error = TimeoutError("timed out")
            raise error

    class _Up:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: object, **kwargs: object) -> str:
            del messages, kwargs
            self.calls += 1
            return "ok"

    first = _Down()
    second = _Up()
    slept: list[float] = []
    chat = FallbackChat(
        [Hop("ollama", "qwen3.5:4b", first), Hop("omniroute", "qwen3.5:4b", second)],
        budget=_budget(Path("/tmp/unused-spend.sqlite")),
        gate=AbortGate(),
        sleeper=slept.append,
    )
    assert chat.invoke(["hello"]) == "ok"
    assert first.calls == 2
    assert second.calls == 1
    assert chat.fallback_used is True
    assert chat.provider_name == "omniroute"
    assert slept == [0.05]
    assert len(chat.timeouts) == 2


def test_http_500_retries_twice_before_the_next_hop() -> None:
    class _Busy:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: object, **kwargs: object) -> object:
            del messages, kwargs
            self.calls += 1
            error = RuntimeError("unavailable")
            error.status_code = 500  # type: ignore[attr-defined]
            raise error

    class _Up:
        def invoke(self, messages: object, **kwargs: object) -> str:
            del messages, kwargs
            return "ok"

    first = _Busy()
    slept: list[float] = []
    chat = FallbackChat(
        [Hop("ollama", "qwen3.5:4b", first), Hop("omniroute", "qwen3.5:4b", _Up())],
        budget=_budget(Path("/tmp/unused-spend.sqlite")),
        gate=AbortGate(),
        sleeper=slept.append,
    )
    assert chat.invoke(["hello"]) == "ok"
    assert first.calls == 3
    assert slept == [0.05, 0.1]
    assert chat.timeouts == []


def test_billing_does_not_call_the_next_provider() -> None:
    class _Credit:
        def invoke(self, messages: object, **kwargs: object) -> object:
            del messages, kwargs
            error = RuntimeError("credit balance is too low")
            error.status_code = 400  # type: ignore[attr-defined]
            raise error

    class _Next:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: object, **kwargs: object) -> object:
            del messages, kwargs
            self.calls += 1
            return "nope"

    nxt = _Next()
    chat = FallbackChat(
        [Hop("anthropic", "claude-haiku", _Credit()), Hop("ollama", "qwen3.5:4b", nxt)],
        budget=_budget(Path("/tmp/unused-spend-2.sqlite")),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(RunAborted, match="BILLING"):
        chat.invoke(["hello"])
    assert nxt.calls == 0
    assert chat.gate.reason is not None


def test_paid_response_without_usage_aborts(tmp_path: Path) -> None:
    class _Silent:
        def invoke(self, messages: object, **kwargs: object) -> str:
            del messages, kwargs
            return "no-usage"

    prices = {
        "claude-haiku": ModelPrice(
            input_per_million=Decimal("1"),
            output_per_million=Decimal("1"),
        )
    }
    chat = FallbackChat(
        [Hop("anthropic", "claude-haiku", _Silent())],
        budget=Budget(
            prices=prices,
            max_run=Decimal("1"),
            max_day=Decimal("1"),
            store=SpendStore(tmp_path / "spend.sqlite"),
            allow_paid=True,
        ),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(RunAborted, match="no token usage"):
        chat.invoke(["hello"])


def test_recorded_anthropic_usage_is_not_zero() -> None:
    fixture = {
        "usage_metadata": {"input_tokens": 0, "output_tokens": 0},
        "response_metadata": {"usage": {"input_tokens": 1152, "output_tokens": 397}},
    }
    assert read_usage(fixture) == (1152, 397)


def _budget(path: Path) -> Budget:
    path.parent.mkdir(parents=True, exist_ok=True)
    return Budget(
        prices={},
        max_run=Decimal("1"),
        max_day=Decimal("1"),
        store=SpendStore(path),
        allow_paid=False,
    )
