"""Per-node model router. The default chain is local, then OmniRoute.

Paid providers are absent from that default. They run only when
``MERIDIAN_ALLOW_PAID=1`` and a USD cap is configured. OmniRoute model ids must
name one model: ``auto`` and combo names are rejected so a benchmark can be repeated.
"""

from collections.abc import Callable
from typing import Any

from pydantic import SecretStr

from meridian.config import MeridianConfigError, Settings
from meridian.llm.budget import PAID, Budget, BudgetError, open_budget
from meridian.llm.errors import AbortGate, ProviderFailure, RunAborted, classify
from meridian.llm.prompt_size import estimate_tokens
from meridian.llm.usage import read_usage

_PROVIDERS = frozenset({"ollama", "omniroute", "anthropic", "xai", "fake"})
_NODES = ("triage", "rootcause", "fix", "critic")
_DEFAULT_CHAIN = "ollama,omniroute"


class Hop:
    """One provider and model the router may call."""

    def __init__(self, provider: str, model: str, client: Any) -> None:
        self.provider = provider
        self.model = model
        self.client = client


class FallbackChat:
    """Try each hop. Transient errors retry twice, then the next hop is used."""

    def __init__(
        self,
        hops: list[Hop],
        *,
        budget: Budget,
        gate: AbortGate,
        sleeper: Callable[[float], None] | None = None,
        backoff_s: float = 0.05,
    ) -> None:
        if not hops:
            raise MeridianConfigError("LLM", ["a provider chain"])
        self.hops = hops
        self.budget = budget
        self.gate = gate
        self.sleeper = _sleep if sleeper is None else sleeper
        self.backoff_s = backoff_s
        self.provider_name = hops[0].provider
        self.active_model = hops[0].model
        self.fallback_used = False
        self.abort_gate = gate

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "FallbackBound":
        return FallbackBound(self, lambda hop: hop.client.bind_tools(tools, **kwargs))

    def with_structured_output(self, schema: type[Any], **kwargs: Any) -> "FallbackBound":
        return FallbackBound(
            self,
            lambda hop: hop.client.with_structured_output(schema, **kwargs),
        )

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._call(messages, lambda hop: hop.client.invoke(messages, **kwargs))

    def _call(self, messages: Any, action: Callable[[Hop], Any]) -> Any:
        self.gate.check()
        last: BaseException | None = None
        for index, hop in enumerate(self.hops):
            for attempt in range(3):
                self.gate.check()
                try:
                    self.budget.preflight(hop.provider, hop.model, _estimate(messages))
                    result = action(hop)
                except BudgetError as exc:
                    self.gate.trip(f"BILLING: {exc}")
                    raise RunAborted("BILLING", str(exc)) from exc
                except RunAborted:
                    raise
                except Exception as exc:
                    kind = classify(exc)
                    if kind in (
                        ProviderFailure.BILLING,
                        ProviderFailure.AUTH,
                        ProviderFailure.QUOTA,
                    ):
                        self.gate.trip(f"{kind.value}: {exc}")
                        raise RunAborted(kind.value, str(exc)) from exc
                    last = exc
                    if kind is ProviderFailure.TRANSIENT and attempt < 2:
                        self.sleeper(self.backoff_s * (attempt + 1))
                        continue
                    break
                else:
                    self._account(hop, result, index)
                    return result
        if last is None:
            raise RuntimeError("no provider left in the chain")
        raise last

    def _account(self, hop: Hop, result: object, index: int) -> None:
        self.provider_name = hop.provider
        self.active_model = hop.model
        self.fallback_used = index > 0
        if hop.provider not in PAID:
            return
        usage = read_usage(result)
        if usage is None:
            self.gate.trip(f"BILLING: {hop.provider} returned no token usage")
            raise RunAborted("BILLING", f"{hop.provider} returned no token usage")
        self.budget.commit(hop.provider, hop.model, usage[0], usage[1])


class FallbackBound:
    """A tools or schema binding that still fails over per call."""

    def __init__(self, parent: FallbackChat, bind: Callable[[Hop], Any]) -> None:
        self.parent = parent
        self._bind = bind
        self.provider_name = parent.provider_name
        self.active_model = parent.active_model
        self.fallback_used = parent.fallback_used
        self.abort_gate = parent.gate

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        del kwargs
        result = self.parent._call(messages, lambda hop: self._bind(hop).invoke(messages))
        self.provider_name = self.parent.provider_name
        self.active_model = self.parent.active_model
        self.fallback_used = self.parent.fallback_used
        return result


def explicit_model(model: str) -> str:
    """Reject OmniRoute ``auto`` and combo ids. A benchmark must name one model."""
    text = model.strip()
    lowered = text.lower()
    if lowered == "auto" or lowered.startswith("auto:") or "+" in text or "combo" in lowered:
        raise MeridianConfigError(
            "model id",
            [
                f"{text} is not an explicit model id. "
                "OmniRoute auto and combo names are rejected so a benchmark run names one model."
            ],
        )
    return text


def resolve_chain(
    provider: str,
    model: str,
    fallback: str,
    *,
    allow_paid: bool,
) -> list[tuple[str, str]]:
    """Ordered hops. Paid providers are skipped unless the run explicitly allows them."""
    primary = provider.strip().lower()
    if primary in PAID and not allow_paid:
        raise MeridianConfigError("paid provider", ["MERIDIAN_ALLOW_PAID"])
    names = [primary]
    for item in fallback.split(","):
        name = item.strip().lower()
        if name and name not in names:
            names.append(name)
    hops: list[tuple[str, str]] = []
    for name in names:
        if name not in _PROVIDERS:
            raise MeridianConfigError("LLM", [f"unknown provider {name}"])
        if name in PAID and not allow_paid:
            continue
        chosen = explicit_model(model) if name == "omniroute" else model.strip()
        if chosen == "":
            raise MeridianConfigError("LLM", ["model id"])
        hops.append((name, chosen))
    if not hops:
        raise MeridianConfigError("LLM", ["provider chain"])
    return hops


def build_node_chat(
    settings: Settings,
    node: str,
    *,
    gate: AbortGate | None = None,
    budget: Budget | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> tuple[FallbackChat, str]:
    """Build the chat for one node. The recorded model name is the primary hop."""
    if node not in _NODES:
        raise MeridianConfigError("LLM", [node])
    provider = _provider(settings, node)
    model = _model(settings, node)
    fallback = _fallback(settings, node)
    if model is None:
        raise MeridianConfigError("LLM", [f"MERIDIAN_{node.upper()}_MODEL"])
    hops = resolve_chain(provider, model, fallback, allow_paid=settings.allow_paid)
    _require_paid(settings, hops)
    shared_gate = AbortGate() if gate is None else gate
    shared_budget = _budget(settings) if budget is None else budget
    clients = [Hop(name, chosen, _client(settings, name, chosen, node)) for name, chosen in hops]
    chat = FallbackChat(clients, budget=shared_budget, gate=shared_gate, sleeper=sleeper)
    return chat, hops[0][1]


NodeChat = tuple[FallbackChat, str]


def build_prepared(settings: Settings) -> tuple[NodeChat, NodeChat, NodeChat, NodeChat]:
    """Four node chats that share one abort gate and one budget."""
    gate = AbortGate()
    budget = _budget(settings)
    built = tuple(build_node_chat(settings, node, gate=gate, budget=budget) for node in _NODES)
    return built[0], built[1], built[2], built[3]


def _provider(settings: Settings, node: str) -> str:
    value = getattr(settings, f"{node}_provider")
    if not isinstance(value, str) or value.strip() == "":
        return "ollama"
    return value.strip().lower()


def _model(settings: Settings, node: str) -> str | None:
    value = getattr(settings, f"{node}_model")
    if not isinstance(value, str) or value.strip() == "":
        return None
    return value.strip()


def _fallback(settings: Settings, node: str) -> str:
    value = getattr(settings, f"{node}_fallback")
    if not isinstance(value, str) or value.strip() == "":
        return _DEFAULT_CHAIN
    return value


def _require_paid(settings: Settings, hops: list[tuple[str, str]]) -> None:
    paid = [name for name, _model_name in hops if name in PAID]
    if not paid:
        return
    if not settings.allow_paid:
        raise MeridianConfigError("paid provider", ["MERIDIAN_ALLOW_PAID"])
    missing = [
        name
        for name, present in (
            ("MERIDIAN_MAX_USD_PER_RUN", settings.max_usd_per_run is not None),
            ("MERIDIAN_MAX_USD_PER_DAY", settings.max_usd_per_day is not None),
        )
        if not present
    ]
    if missing:
        raise MeridianConfigError("paid provider", missing)
    from meridian.llm.prices import load_prices

    prices = load_prices()
    unpriced = [model for name, model in hops if name in PAID and model not in prices]
    if unpriced:
        raise MeridianConfigError("price", unpriced)


def _budget(settings: Settings) -> Budget:
    return open_budget(
        spend_path=settings.spend_path,
        allow_paid=settings.allow_paid,
        max_run=settings.max_usd_per_run,
        max_day=settings.max_usd_per_day,
    )


def _client(settings: Settings, provider: str, model: str, node: str) -> Any:
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=settings.ollama_base_url)
    if provider == "omniroute":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model_name=explicit_model(model),
            openai_api_base=settings.omniroute_base_url,
            openai_api_key=SecretStr("omniroute"),
        )
    if provider == "anthropic":
        from meridian.agents.live_llm import build_live_chat

        settings.require_anthropic()
        return build_live_chat(settings, model=model, model_env="MERIDIAN_ANTHROPIC_MODEL")
    if provider == "xai":
        from meridian.agents.live_llm import build_live_critic

        settings.require_xai()
        return build_live_critic(settings, model=model)
    if provider == "fake":
        return _fake(node)
    raise MeridianConfigError("LLM", [provider])


def _fake(node: str) -> Any:
    from meridian.agents.fake_critic import HeuristicCriticModel
    from meridian.agents.fake_fix import HeuristicFixModel
    from meridian.agents.fake_llm import FakeTriageModel
    from meridian.agents.fake_root_cause import HeuristicRootCauseModel

    if node == "critic":
        return HeuristicCriticModel()
    if node == "fix":
        return HeuristicFixModel()
    if node == "rootcause":
        return HeuristicRootCauseModel()
    return FakeTriageModel()


def _estimate(messages: Any) -> int:
    if not isinstance(messages, list):
        return estimate_tokens(str(messages))
    total = 0
    for message in messages:
        content = getattr(message, "content", message)
        total += estimate_tokens(content if isinstance(content, str) else str(content))
    return total


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)
