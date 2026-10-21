"""Capability selection, the cache key, and a refused connection."""

import json
from pathlib import Path

import pytest

from meridian.config import Settings
from meridian.llm.capabilities import (
    Capability,
    _http_probe,
    _ollama_version,
    _probe_ollama,
    _probe_openai,
    _read,
    _request,
    apply_methods,
    arm_node,
    cache_key,
    format_table,
    method_for,
    plan_nodes,
    supported,
)
from meridian.run import PreparedChats, _capabilities


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "triage_model": "qwen3.5:4b",
        "rootcause_model": "qwen3.5:4b",
        "fix_model": "qwen3.5:4b",
        "critic_model": "gemma4:e4b",
        "allow_paid": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_method_follows_tool_then_schema_then_json_mode() -> None:
    base = {"provider": "ollama", "model": "qwen3.5:4b", "ollama_version": "0.40.6"}
    assert method_for(Capability(**base, tool_calling=True, structured_json_schema=True)) == (
        "submit_tool"
    )
    assert method_for(Capability(**base, structured_json_schema=True)) == "json_schema"
    assert method_for(Capability(**base, structured_json_mode=True)) == "json_mode"
    assert method_for(Capability(**base)) is None
    assert supported(200, "{}") is True
    assert supported(501, '{"error":"structured output is unavailable"}') is False
    assert supported(200, "structured output is unavailable") is False
    assert supported(0, "connection refused") is False


def test_plan_caches_by_provider_model_and_version(tmp_path: Path) -> None:
    calls: list[str] = []

    def probe(provider: str, model: str, version: str, settings: Settings) -> Capability:
        del settings
        calls.append(cache_key(provider, model, version))
        return Capability(
            provider=provider,
            model=model,
            ollama_version=version,
            tool_calling=provider == "ollama",
            structured_json_schema=provider == "omniroute",
        )

    path = tmp_path / "capabilities.json"
    settings = _settings()
    rows, methods = plan_nodes(settings, cache_path=path, probe=probe, ollama_version="0.40.6")
    again, _methods = plan_nodes(settings, cache_path=path, probe=probe, ollama_version="0.40.6")
    assert len(again) == len(rows)
    assert calls == list(dict.fromkeys(calls))
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "ollama|qwen3.5:4b|0.40.6" in stored
    assert methods["triage"] == ["submit_tool", "json_schema"]
    assert methods["critic"][0] == "submit_tool"
    table = format_table(rows + rows, methods)
    assert "version=0.40.6" in table
    assert table.count("qwen3.5:4b") == 2
    assert "node triage submit_tool,json_schema" in table


def test_unreachable_probes_are_not_cached(tmp_path: Path) -> None:
    calls: list[int] = []

    def probe(provider: str, model: str, version: str, settings: Settings) -> Capability:
        del settings
        calls.append(1)
        return Capability(provider=provider, model=model, ollama_version=version, reachable=False)

    path = tmp_path / "cap.json"
    local = _settings(
        triage_fallback="ollama",
        rootcause_fallback="ollama",
        fix_fallback="ollama",
        critic_fallback="ollama",
    )
    rows, _methods = plan_nodes(local, cache_path=path, probe=probe, ollama_version="0.40.6")
    plan_nodes(local, cache_path=path, probe=probe, ollama_version="0.40.6")
    assert len(calls) == 8
    assert path.exists() is False
    assert "UNREACHABLE" in format_table(rows, {"triage": [None]})


def test_a_day_old_probe_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "cap.json"
    key = cache_key("ollama", "qwen3.5:4b", "0.40.6")
    stale = Capability(
        provider="ollama",
        model="qwen3.5:4b",
        ollama_version="0.40.6",
        tool_calling=True,
        reachable=True,
        probed_at="2020-01-01T00:00:00+00:00",
    )
    path.write_text(json.dumps({key: stale.model_dump()}), encoding="utf-8")
    seen: list[str] = []

    def probe(provider: str, model: str, version: str, settings: Settings) -> Capability:
        del settings
        seen.append(cache_key(provider, model, version))
        return Capability(
            provider=provider,
            model=model,
            ollama_version=version,
            structured_json_mode=True,
            reachable=True,
        )

    plan_nodes(
        _settings(triage_fallback="ollama"),
        cache_path=path,
        probe=probe,
        ollama_version="0.40.6",
    )
    assert key in seen
    plan_nodes(
        _settings(triage_fallback="ollama"),
        cache_path=path,
        probe=probe,
        ollama_version="9",
    )
    assert cache_key("ollama", "qwen3.5:4b", "9") in seen


def test_a_401_is_auth_failed_and_is_not_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import SecretStr

    headers: list[dict[str, str]] = []

    def request(
        method: str,
        url: str,
        payload: dict[str, object] | None,
        extra: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        del method, url, payload
        headers.append(dict(extra or {}))
        return 401, "unauthorized"

    monkeypatch.setattr("meridian.llm.capabilities._request", request)
    settings = _settings(omniroute_api_key=SecretStr("probe-token"))
    row = _probe_openai(
        "http://127.0.0.1:9/v1",
        "omniroute",
        "gemini/gemini-2.5-flash",
        "",
        settings,
    )
    assert row.reachable is True
    assert row.auth_failed is True
    assert row.tool_calling is False
    assert method_for(row) is None
    assert "method=AUTH_FAILED" in format_table([row], {"triage": [None]})
    assert headers[0].get("Authorization") == "Bearer probe-token"
    calls: list[int] = []

    def probe(provider: str, model: str, version: str, chosen: Settings) -> Capability:
        del chosen
        calls.append(1)
        return Capability(
            provider=provider,
            model=model,
            ollama_version=version,
            reachable=True,
            auth_failed=True,
        )

    path = tmp_path / "cap.json"
    local = _settings(
        triage_fallback="ollama",
        rootcause_fallback="ollama",
        fix_fallback="ollama",
        critic_fallback="ollama",
    )
    plan_nodes(local, cache_path=path, probe=probe, ollama_version="0.40.6")
    plan_nodes(local, cache_path=path, probe=probe, ollama_version="0.40.6")
    assert len(calls) == 8
    assert path.exists() is False


def test_a_connection_error_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "meridian.llm.capabilities._request",
        lambda *_args, **_kwargs: (0, "timed out"),
    )
    row = _probe_ollama("http://localhost:11434", "qwen3.5:4b", "0.40.6")
    assert row.reachable is False
    assert "method=UNREACHABLE" in format_table([row], {"triage": [None]})
    answered = _probe_openai("http://localhost:9/v1", "omniroute", "m", "")
    assert answered.reachable is False


def test_a_chain_with_no_method_is_named(tmp_path: Path) -> None:
    def probe(provider: str, model: str, version: str, settings: Settings) -> Capability:
        del provider, model, version, settings
        return Capability(provider="ollama", model="qwen3.5:4b", ollama_version="")

    _rows, methods = plan_nodes(
        _settings(triage_fallback="ollama"),
        cache_path=tmp_path / "cap.json",
        probe=probe,
        ollama_version="",
    )
    assert methods["triage"] == [None]


def test_arm_node_rejects_an_empty_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "meridian.llm.capabilities.plan_nodes",
        lambda settings: ([], {"triage": [None]}),
    )
    with pytest.raises(RuntimeError, match="no structured method"):
        arm_node(_settings(), "triage", type("Chat", (), {})())
    monkeypatch.setattr(
        "meridian.llm.capabilities.plan_nodes",
        lambda settings: ([], {"triage": ["json_mode"]}),
    )
    chat = type("Chat", (), {})()
    arm_node(_settings(), "triage", chat)
    assert chat.output_method == "json_mode"


def test_apply_methods_keeps_the_first_usable_hop() -> None:
    chat = type("Chat", (), {})()
    apply_methods(chat, [None, "json_mode"])
    assert chat.hop_methods == [None, "json_mode"]
    assert chat.output_method == "json_mode"


def test_probe_helpers_do_not_open_a_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(
        method: str,
        url: str,
        payload: dict[str, object] | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        del method, headers
        if payload is None:
            return 200, '{"version": "0.40.6"}'
        if "tools" in payload:
            return 200, "{}"
        body = payload.get("format", payload.get("response_format"))
        if body == "json" or (isinstance(body, dict) and body.get("type") == "json_object"):
            return 501, '{"error":"structured output is unavailable"}'
        return 501, '{"error":"structured output is unavailable"}'

    monkeypatch.setattr("meridian.llm.capabilities._request", fake)
    ollama = _probe_ollama("http://localhost:11434", "qwen3.5:4b", "0.40.6")
    assert ollama.tool_calling is True
    assert ollama.structured_json_schema is False
    assert ollama.structured_json_mode is False
    omni = _probe_openai("http://localhost:20128/v1", "omniroute", "qwen3.5:4b", "")
    assert omni.tool_calling is True
    assert omni.structured_json_mode is False
    assert _ollama_version("http://localhost:11434") == "0.40.6"
    assert _http_probe("ollama", "qwen3.5:4b", "0.40.6", _settings()).tool_calling is True
    assert _http_probe("omniroute", "qwen3.5:4b", "", _settings()).tool_calling is True
    assert _http_probe("xai", "grok", "", _settings()).structured_json_schema is True
    monkeypatch.setattr(
        "meridian.llm.capabilities._request",
        lambda method, url, payload: (200, "not-json"),
    )
    assert _ollama_version("http://localhost:11434") == ""
    monkeypatch.setattr(
        "meridian.llm.capabilities._request",
        lambda method, url, payload: (503, ""),
    )
    assert _ollama_version("http://localhost:11434") == ""
    monkeypatch.setattr(
        "meridian.llm.capabilities._request",
        lambda method, url, payload: (200, '{"version": 1}'),
    )
    assert _ollama_version("http://localhost:11434") == ""
    paid = _http_probe("anthropic", "claude-haiku", "", _settings())
    assert paid.tool_calling is True
    assert paid.structured_json_schema is True
    blank = _http_probe("fake", "fake", "", _settings())
    assert blank.tool_calling is False


def test_live_probe_uses_the_version_and_skips_a_blank_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(
        method: str,
        url: str,
        payload: dict[str, object] | None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        del method, url, payload, headers
        return 200, "{}"

    monkeypatch.setattr("meridian.llm.capabilities._request", fake)
    monkeypatch.setattr("meridian.llm.capabilities._ollama_version", lambda _url: "9")
    settings = _settings(triage_model=None, triage_fallback="ollama")
    _rows, methods = plan_nodes(settings, cache_path=tmp_path / "cap.json")
    assert methods["triage"] == []
    assert methods["critic"][0] == "submit_tool"

    def as_dict(provider: str, model: str, version: str, settings: Settings) -> dict[str, object]:
        del settings
        return {
            "provider": provider,
            "model": model,
            "ollama_version": version,
            "tool_calling": True,
        }

    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    rows, _methods = plan_nodes(
        _settings(triage_fallback="ollama"),
        cache_path=listed,
        probe=as_dict,
    )
    assert rows[0].tool_calling is True
    assert _read(listed) != {}


def test_request_reads_a_local_status() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            raw = b'{"version":"0"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            raw = b'{"error":"structured output is unavailable"}'
            self.send_response(501)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt: str, *args: object) -> None:
            del fmt, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request("GET", f"http://127.0.0.1:{port}/api/version", None)
        missed, text = _request("POST", f"http://127.0.0.1:{port}/api/chat", {"model": "qwen"})
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200
    assert "version" in body
    assert missed == 501
    assert "unavailable" in text


def test_request_reports_a_closed_port_and_a_bad_cache(tmp_path: Path) -> None:
    status, body = _request("GET", "http://127.0.0.1:1/nope", None)
    assert status == 0
    assert body
    broken = tmp_path / "capabilities.json"
    broken.write_text("{", encoding="utf-8")
    assert _read(broken) == {}
    assert _read(tmp_path / "missing.json") == {}


def test_capabilities_abort_and_apply(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chats = PreparedChats(
        type("C", (), {})(),
        "qwen",
        type("C", (), {})(),
        "qwen",
        type("C", (), {})(),
        "qwen",
        type("C", (), {})(),
        "gemma",
    )

    def none(settings: Settings) -> tuple[list[Capability], dict[str, list[str | None]]]:
        del settings
        row = Capability(provider="ollama", model="qwen3.5:4b", ollama_version="0.40.6")
        methods = {node: [None] for node in ("triage", "rootcause", "fix", "critic")}
        return [row], methods

    monkeypatch.setattr("meridian.run.plan_nodes", none)
    reason = _capabilities(_settings(), chats)
    assert reason is not None
    assert "triage" in reason
    assert "method=none" in capsys.readouterr().err

    def ready(settings: Settings) -> tuple[list[Capability], dict[str, list[str | None]]]:
        del settings
        row = Capability(
            provider="ollama",
            model="qwen3.5:4b",
            ollama_version="0.40.6",
            tool_calling=True,
        )
        methods = {node: ["submit_tool"] for node in ("triage", "rootcause", "fix", "critic")}
        return [row], methods

    monkeypatch.setattr("meridian.run.plan_nodes", ready)
    assert _capabilities(_settings(), chats) is None
    assert chats.critic_chat.output_method == "submit_tool"
