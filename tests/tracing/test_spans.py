"""A fake run produces the span tree and never records a prompt or a secret."""

from types import SimpleNamespace

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from meridian.tracing import (
    _category,
    _install,
    case_parent,
    finish_case,
    install_memory,
    note_llm,
    note_sandbox,
    note_tool,
    reset,
    run_span,
    span,
)


def test_span_tree_has_no_prompt_or_secret() -> None:
    exporter = InMemorySpanExporter()
    install_memory(exporter)
    try:
        _tree(exporter)
    finally:
        reset()


def _tree(exporter: InMemorySpanExporter) -> None:
    case = SimpleNamespace(
        verification=SimpleNamespace(passed=True),
        needs_human=False,
        patch=object(),
        critic_rounds=1,
        fix_attempts=1,
    )
    with run_span("run-1"):
        parent = case_parent("run-1", "case-1")
        with span("meridian.node", parent=parent) as node:
            node.set_attribute("node", "fix")
            node.set_attribute("prompt", "full prompt text")
            node.set_attribute("api_key", "sk-secret")
            note_llm(
                provider="ollama",
                model="heuristic",
                prompt_version="v1",
                input_tokens=3,
                output_tokens=4,
                latency_ms=5,
                timed_out=False,
                think=False,
                attempt=1,
            )
            note_tool("list_events", 2)
            note_sandbox("repair", "ok", 3, 3)
        finish_case("run-1", "case-1", {"cases": {"case-1": case}})
    finished = exporter.get_finished_spans()
    names = {item.name for item in finished}
    assert {
        "meridian.run",
        "meridian.case",
        "meridian.node",
        "meridian.llm",
        "meridian.tool",
        "meridian.sandbox",
    } <= names
    assert len({item.context.trace_id for item in finished}) == 1
    outcomes = [item.attributes.get("outcome") for item in finished if item.name == "meridian.case"]
    assert outcomes == ["VERIFIED_AWAITING_APPROVAL"]
    for item in finished:
        attributes = item.attributes or {}
        assert "prompt" not in attributes
        assert "api_key" not in attributes
        assert "sk-secret" not in attributes.values()
        assert "full prompt text" not in attributes.values()


def test_nested_run_reuses_the_open_span() -> None:
    exporter = InMemorySpanExporter()
    install_memory(exporter)
    try:
        with run_span("run-1"), run_span("run-1"):
            case_parent("run-1", "-")
        names = [item.name for item in exporter.get_finished_spans()]
        assert names == ["meridian.run"]
    finally:
        reset()


def test_disabled_tracer_and_outcome_categories() -> None:
    reset()
    assert case_parent("run-1", "case-1").name == ""
    finish_case("run-1", "missing", {})
    escalated = SimpleNamespace(verification=None, needs_human=True, patch=object())
    bare = SimpleNamespace(verification=None, needs_human=False, patch=None)
    critic = SimpleNamespace(
        verification=None, needs_human=False, patch=object(), critic_rounds=3, fix_attempts=1
    )
    limited = SimpleNamespace(
        verification=None, needs_human=False, patch=object(), critic_rounds=1, fix_attempts=3
    )
    other = SimpleNamespace(
        verification=None, needs_human=False, patch=object(), critic_rounds=1, fix_attempts=1
    )
    assert _category({"cases": {"a": escalated}}) == "AGENT_ESCALATED"
    assert _category({"cases": {"a": bare}}) == "NO_PATCH"
    assert _category({"cases": {"a": critic}}) == "CRITIC_LIMIT"
    assert _category({"cases": {"a": limited}}) == "FIX_ATTEMPT_LIMIT"
    assert _category({"cases": {"a": other}}) == "OTHER"
    assert _category({}) == "OTHER"


def test_console_and_otlp_exporters_install() -> None:
    try:
        _install(None, console=True, endpoint="")
        with span("meridian.node") as opened:
            opened.set_attribute("token", "hidden")
            opened.set_attribute("bot_token", "hidden")
            assert "token" not in opened.attributes
            assert "bot_token" not in opened.attributes
        _install(None, console=False, endpoint="http://127.0.0.1:9")
    finally:
        reset()
