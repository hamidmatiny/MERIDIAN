"""Construct local clients and exercise fallback, paid, and abort branches."""

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from meridian.config import MeridianConfigError, Settings
from meridian.llm.budget import Budget, ModelPrice, SpendStore
from meridian.llm.errors import AbortGate, RunAborted
from meridian.llm.prices import load_prices
from meridian.llm.router import (
    FallbackChat,
    Hop,
    _client,
    _fallback,
    _model,
    _provider,
    build_node_chat,
    build_prepared,
    resolve_chain,
)


class _Ok:
    def __init__(self, payload: object = "ok") -> None:
        self.payload = payload
        self.calls = 0

    def bind_tools(self, tools: object, **kwargs: object) -> "_Ok":
        del tools, kwargs
        return self

    def with_structured_output(self, schema: object, **kwargs: object) -> "_Ok":
        del schema, kwargs
        return self

    def invoke(self, messages: object, **kwargs: object) -> object:
        del messages, kwargs
        self.calls += 1
        return self.payload


class _Raise:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def invoke(self, messages: object, **kwargs: object) -> object:
        del messages, kwargs
        raise self.error


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "fix_model": "qwen3.5:4b",
        "critic_model": "gemma4:e4b",
        "spend_path": str(tmp_path / "spend.sqlite"),
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _budget(tmp_path: Path, *, allow_paid: bool = False) -> Budget:
    price = ModelPrice(input_per_million=Decimal("1"), output_per_million=Decimal("2"))
    return Budget(
        prices={"claude-haiku": price, "grok-4.7": price},
        max_run=Decimal("1"),
        max_day=Decimal("1"),
        store=SpendStore(tmp_path / "ledger.sqlite"),
        allow_paid=allow_paid,
    )


def test_local_and_fake_clients_construct(tmp_path: Path) -> None:
    local = _settings(
        tmp_path,
        triage_provider="ollama",
        triage_model="qwen3.5:4b",
        triage_fallback="ollama,omniroute",
    )
    chat, name = build_node_chat(local, "triage", sleeper=lambda _seconds: None)
    assert name == "qwen3.5:4b"
    assert type(chat.hops[0].client).__name__ == "ChatOllama"
    assert type(chat.hops[1].client).__name__ == "ChatOpenAI"
    fake = _settings(
        tmp_path,
        triage_provider="fake",
        triage_model="fake-triage",
        triage_fallback="fake",
        rootcause_provider="fake",
        rootcause_model="fake-root",
        rootcause_fallback="fake",
        fix_provider="fake",
        fix_model="configured-fix",
        fix_fallback="fake",
        critic_provider="fake",
        critic_model="configured-critic",
        critic_fallback="fake",
    )
    built = build_prepared(fake)
    assert [item[1] for item in built] == [
        "fake-triage",
        "fake-root",
        "configured-fix",
        "configured-critic",
    ]
    assert type(built[1][0].hops[0].client).__name__ == "HeuristicRootCauseModel"
    assert type(built[2][0].hops[0].client).__name__ == "HeuristicFixModel"
    assert type(built[3][0].hops[0].client).__name__ == "HeuristicCriticModel"


def test_paid_clients_require_a_price(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        allow_paid=True,
        anthropic_api_key=SecretStr("test-key"),
        xai_api_key=SecretStr("xai-test"),
        triage_provider="anthropic",
        triage_model="claude-haiku",
        triage_fallback="anthropic",
    )
    with pytest.raises(MeridianConfigError, match="MERIDIAN_MAX_USD"):
        build_node_chat(settings, "triage")
    capped = _settings(
        tmp_path,
        allow_paid=True,
        max_usd_per_run=0.25,
        max_usd_per_day=1.0,
        anthropic_api_key=SecretStr("test-key"),
        xai_api_key=SecretStr("xai-test"),
        triage_provider="anthropic",
        triage_model="claude-haiku",
        triage_fallback="anthropic",
    )
    with pytest.raises(MeridianConfigError, match="price"):
        build_node_chat(capped, "triage")
    assert "claude-haiku" not in load_prices()
    anthropic = _client(capped, "anthropic", "claude-haiku", "triage")
    grok = _client(capped, "xai", "grok-4.7", "critic")
    assert type(anthropic).__name__ == "ChatAnthropic"
    assert type(grok).__name__ == "ChatXAI"
    with pytest.raises(MeridianConfigError, match="nope"):
        _client(capped, "nope", "model", "triage")


def test_chain_rejects_unknown_and_blank_models() -> None:
    with pytest.raises(MeridianConfigError, match="unknown provider"):
        resolve_chain("ollama", "qwen3.5:4b", "ollama,mystery", allow_paid=False)
    with pytest.raises(MeridianConfigError, match="model id"):
        resolve_chain("ollama", "  ", "ollama", allow_paid=False)
    with pytest.raises(MeridianConfigError, match="explicit"):
        resolve_chain("omniroute", "auto:best", "omniroute", allow_paid=False)
    with pytest.raises(MeridianConfigError, match="a provider chain"):
        FallbackChat([], budget=_budget(Path("/tmp")), gate=AbortGate())


def test_blank_node_fields_use_the_free_default() -> None:
    class _Blank:
        triage_provider = "   "
        triage_model = " "
        triage_fallback = ""

    assert _provider(_Blank(), "triage") == "ollama"
    assert _model(_Blank(), "triage") is None
    assert _fallback(_Blank(), "triage") == "ollama,omniroute"


def test_build_rejects_an_unknown_node_and_a_missing_model(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(MeridianConfigError, match="sentinel"):
        build_node_chat(settings, "sentinel")
    with pytest.raises(MeridianConfigError, match="MERIDIAN_TRIAGE_MODEL"):
        build_node_chat(settings, "triage")


def test_billing_auth_and_quota_do_not_fall_through(tmp_path: Path) -> None:
    budget = _budget(tmp_path, allow_paid=True)
    gate = AbortGate()
    credit = RuntimeError("credit balance is too low")
    credit.status_code = 400
    chat = FallbackChat(
        [Hop("anthropic", "claude-haiku", _Raise(credit)), Hop("ollama", "qwen", _Ok())],
        budget=budget,
        gate=gate,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(RunAborted, match="BILLING"):
        chat.invoke("prompt")
    assert gate.reason is not None
    auth = RuntimeError("denied")
    auth.status_code = 401
    with pytest.raises(RunAborted, match="AUTH"):
        FallbackChat(
            [Hop("anthropic", "claude-haiku", _Raise(auth))],
            budget=_budget(tmp_path, allow_paid=True),
            gate=AbortGate(),
            sleeper=lambda _seconds: None,
        ).invoke([object()])
    limited = RuntimeError("quota exceeded")
    limited.status_code = 429
    with pytest.raises(RunAborted, match="QUOTA"):
        FallbackChat(
            [Hop("xai", "grok-4.7", _Raise(limited))],
            budget=_budget(tmp_path, allow_paid=True),
            gate=AbortGate(),
            sleeper=lambda _seconds: None,
        ).invoke("prompt")


def test_bad_output_falls_back_and_a_bound_call_records_the_hop(tmp_path: Path) -> None:
    budget = _budget(tmp_path)
    second = _Ok("kept")
    hops = [
        Hop("ollama", "qwen3.5:4b", _Raise(ValueError("bad json"))),
        Hop("omniroute", "qwen", second),
    ]
    chat = FallbackChat(
        hops,
        budget=budget,
        gate=AbortGate(),
        sleeper=lambda _seconds: (_ for _ in ()).throw(AssertionError("no retry")),
    )
    bound = chat.with_structured_output(dict)
    assert bound.invoke("prompt") == "kept"
    assert bound.fallback_used is True
    assert bound.provider_name == "omniroute"
    tools = chat.bind_tools([])
    assert tools.invoke([{"content": "again"}]) == "kept"


def test_a_paid_response_without_usage_aborts(tmp_path: Path) -> None:
    chat = FallbackChat(
        [Hop("anthropic", "claude-haiku", _Ok(object()))],
        budget=_budget(tmp_path, allow_paid=True),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(RunAborted, match="no token usage"):
        chat.invoke("prompt")


def test_a_paid_response_commits_usage(tmp_path: Path) -> None:
    budget = _budget(tmp_path, allow_paid=True)
    payload = {"usage_metadata": {"input_tokens": 20, "output_tokens": 5}}
    chat = FallbackChat(
        [Hop("anthropic", "claude-haiku", _Ok(payload))],
        budget=budget,
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
    )
    assert chat.invoke("prompt") == payload
    assert budget.run_spent > 0


def test_budget_and_an_open_abort_stop_the_chain(tmp_path: Path) -> None:
    price = ModelPrice(input_per_million=Decimal("3"), output_per_million=Decimal("1"))
    tiny = Budget(
        prices={"claude-haiku": price},
        max_run=Decimal("0"),
        max_day=Decimal("1"),
        store=SpendStore(tmp_path / "tiny.sqlite"),
        allow_paid=True,
    )
    with pytest.raises(RunAborted, match="BILLING"):
        FallbackChat(
            [Hop("anthropic", "claude-haiku", _Ok())],
            budget=tiny,
            gate=AbortGate(),
            sleeper=lambda _seconds: None,
        ).invoke("this prompt is not free")
    gate = AbortGate()
    gate.trip("AUTH: denied")
    with pytest.raises(RunAborted, match="ABORTED"):
        FallbackChat(
            [Hop("ollama", "qwen", _Ok())],
            budget=_budget(tmp_path),
            gate=gate,
            sleeper=lambda _seconds: None,
        ).invoke("prompt")


def test_an_inner_abort_is_not_classified_again(tmp_path: Path) -> None:
    with pytest.raises(RunAborted, match="QUOTA"):
        FallbackChat(
            [Hop("ollama", "qwen", _Raise(RunAborted("QUOTA", "limit")))],
            budget=_budget(tmp_path),
            gate=AbortGate(),
            sleeper=lambda _seconds: None,
        ).invoke("prompt")


def test_provider_and_paid_flag_edges() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, triage_provider="nope")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, triage_provider=1)
    allowed = Settings(_env_file=None, allow_paid="yes")
    assert allowed.allow_paid is True
    with pytest.raises(ValidationError):
        Settings(_env_file=None, allow_paid=2)
