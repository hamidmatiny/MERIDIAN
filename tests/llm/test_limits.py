"""Call limits: one timeout retry, load time, and the combined model cap."""

import time
from pathlib import Path

import pytest

from meridian.config import Settings
from meridian.llm.budget import Budget, SpendStore
from meridian.llm.errors import AbortGate, RunAborted
from meridian.llm.resident import budget_problem, configured_models, parse_ollama_list, size_report
from meridian.llm.router import FallbackChat, Hop
from meridian.llm.usage import TimeoutNote, assemble_calls


def _budget(tmp_path: Path) -> Budget:
    return Budget(
        prices={},
        max_run=None,
        max_day=None,
        store=SpendStore(tmp_path / "ledger.sqlite"),
        allow_paid=False,
    )


class _Message:
    def __init__(self) -> None:
        self.response_metadata = {"load_duration": 2_000_000_000}


def test_assemble_keeps_the_timeout_and_the_load_time(tmp_path: Path) -> None:
    chat = FallbackChat(
        [Hop("ollama", "qwen3.5:4b", object())],
        budget=_budget(tmp_path),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
    )
    chat.timeouts.append(TimeoutNote("ollama", "qwen3.5:4b", 180000))
    calls = assemble_calls(
        node="fix",
        model="qwen3.5:4b",
        prompt_version="v1",
        raw=_Message(),
        latency_ms=12,
        attempt=1,
        succeeded=True,
        bound=chat,
        tokens_before=0,
        tokens_after=0,
    )
    assert calls[0].timed_out is True
    assert calls[0].succeeded is False
    assert calls[1].timed_out is False
    assert calls[1].load_duration_ms == 2000
    assert calls[1].attempt == 2
    assert chat.timeouts == []


def test_a_slow_provider_is_cut_off_and_recorded_as_a_timeout(tmp_path: Path) -> None:
    class _Slow:
        def invoke(self, messages: object, **kwargs: object) -> str:
            del messages, kwargs
            time.sleep(2)
            return "late"

    chat = FallbackChat(
        [Hop("ollama", "qwen3.5:4b", _Slow())],
        budget=_budget(tmp_path),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
        call_timeout_s=0.05,
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        chat.invoke(["hello"])
    assert time.monotonic() - started < 1.0
    assert len(chat.timeouts) == 2
    calls = assemble_calls(
        node="fix",
        model="qwen3.5:4b",
        prompt_version="v1",
        raw=None,
        latency_ms=0,
        attempt=1,
        succeeded=False,
        bound=chat,
        tokens_before=0,
        tokens_after=0,
    )
    timed = [item for item in calls if item.timed_out]
    assert len(timed) == 2
    assert all(item.succeeded is False for item in timed)


def test_a_passed_deadline_aborts_before_the_model(tmp_path: Path) -> None:
    class _Up:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: object, **kwargs: object) -> str:
            del messages, kwargs
            self.calls += 1
            return "ok"

    client = _Up()
    chat = FallbackChat(
        [Hop("ollama", "qwen3.5:4b", client)],
        budget=_budget(tmp_path),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
        deadline=0,
    )
    with pytest.raises(RunAborted) as caught:
        chat.invoke(["hello"])
    assert caught.value.kind == "WALL_BUDGET"
    assert client.calls == 0


def test_models_over_the_cap_name_the_largest() -> None:
    text = "\n".join(
        [
            "NAME ID SIZE MODIFIED",
            "qwen3.5:27b-q4_K_M abc 17 GB 2 days ago",
            "gemma4:12b def 7.7 GB yesterday",
            "qwen3.5:4b ghi 4.0 GB yesterday",
        ]
    )
    sizes = parse_ollama_list(text)
    settings = Settings(
        _env_file=None,
        triage_model="qwen3.5:4b",
        rootcause_model="qwen3.5:27b-q4_K_M",
        fix_model="qwen3.5:27b-q4_K_M",
        critic_model="gemma4:12b",
        max_model_gb=26,
    )
    problem = budget_problem(settings, sizes)
    assert problem is not None
    assert "28.7 GB" in problem
    assert "Downsize qwen3.5:27b-q4_K_M" in problem
    assert budget_problem(settings.model_copy(update={"max_model_gb": 30}), sizes) is None
    missing = budget_problem(settings, {"qwen3.5:4b": 4.0})
    assert missing is not None
    assert missing.startswith("cannot read model size for")


def test_a_tag_shared_by_three_nodes_is_counted_once() -> None:
    settings = Settings(
        _env_file=None,
        triage_model="qwen3.5:27b-q4_K_M",
        rootcause_model="qwen3.5:27b-q4_K_M",
        fix_model="qwen3.5:27b-q4_K_M",
        critic_model="gemma4:12b",
        max_model_gb=26,
    )
    sizes = {"qwen3.5:27b-q4_K_M": 17.0, "gemma4:12b": 7.7, "qwen3.5:4b": 4.0}
    assert configured_models(settings) == ["qwen3.5:27b-q4_K_M", "gemma4:12b"]
    assert "24.7 GB" in size_report(settings, sizes)
    assert budget_problem(settings, sizes) is None
