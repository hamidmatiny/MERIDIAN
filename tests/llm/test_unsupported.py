"""A 501 moves the node to the next provider and is not swallowed."""

import json
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from meridian.llm.budget import Budget, SpendStore
from meridian.llm.errors import AbortGate, ProviderFailure, RunAborted, call_model, classify
from meridian.llm.router import FallbackChat, Hop


class _Down:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: object, **kwargs: object) -> object:
        del messages, kwargs
        self.calls += 1
        error = RuntimeError("structured output is unavailable")
        error.status_code = 501  # type: ignore[attr-defined]
        raise error


class _Up:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: object, **kwargs: object) -> str:
        del messages, kwargs
        self.calls += 1
        return "ok"


def _budget(path: Path) -> Budget:
    path.parent.mkdir(parents=True, exist_ok=True)
    return Budget(
        prices={},
        max_run=Decimal("1"),
        max_day=Decimal("1"),
        store=SpendStore(path),
        allow_paid=False,
    )


def test_501_is_unsupported_and_is_not_retried(tmp_path: Path) -> None:
    missing = RuntimeError("structured output is unavailable")
    missing.status_code = 501  # type: ignore[attr-defined]
    assert classify(missing) is ProviderFailure.UNSUPPORTED
    assert classify(RuntimeError("model does not support tools")) is ProviderFailure.UNSUPPORTED
    first = _Down()
    second = _Up()
    slept: list[float] = []
    chat = FallbackChat(
        [Hop("ollama", "qwen3.5:4b", first), Hop("omniroute", "qwen3.5:4b", second)],
        budget=_budget(tmp_path / "spend.sqlite"),
        gate=AbortGate(),
        sleeper=slept.append,
    )
    assert chat.invoke(["hello"]) == "ok"
    assert first.calls == 1
    assert second.calls == 1
    assert chat.fallback_used is True
    assert chat.provider_name == "omniroute"
    assert slept == []


def test_an_exhausted_unsupported_chain_aborts(tmp_path: Path) -> None:
    chat = FallbackChat(
        [Hop("ollama", "qwen3.5:4b", _Down())],
        budget=_budget(tmp_path / "spend.sqlite"),
        gate=AbortGate(),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(RunAborted, match="UNSUPPORTED") as caught:
        call_model(chat, [HumanMessage(content="hi")])
    assert caught.value.kind == "UNSUPPORTED"
    assert chat.abort_gate.reason is not None


_PATHS: list[str] = []


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        _PATHS.append(self.path)
        if self.path.startswith("/api/"):
            raw = b'{"error":"structured output is unavailable"}'
            status = 501
        else:
            raw = json.dumps(
                {
                    "id": "x",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            status = 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: object) -> None:
        del fmt, args


def test_ollama_501_falls_back_to_the_configured_omniroute(tmp_path: Path) -> None:
    _PATHS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ollama = ChatOllama(
            model="qwen3.5:4b",
            base_url=f"http://127.0.0.1:{port}",
            client_kwargs={"timeout": 5},
        )
        omni = ChatOpenAI(
            model_name="qwen3.5:4b",
            openai_api_base=f"http://127.0.0.1:{port}/v1",
            openai_api_key=SecretStr("omniroute"),
            max_retries=0,
            request_timeout=5,
        )
        chat = FallbackChat(
            [Hop("ollama", "qwen3.5:4b", ollama), Hop("omniroute", "qwen3.5:4b", omni)],
            budget=_budget(tmp_path / "spend.sqlite"),
            gate=AbortGate(),
            sleeper=lambda _seconds: None,
        )
        result = chat.invoke([HumanMessage(content="hi")])
    finally:
        server.shutdown()
        server.server_close()
    content = getattr(result, "content", "")
    assert content == "ok"
    assert chat.fallback_used is True
    assert any(path.startswith("/api/") for path in _PATHS)
    assert any(path.endswith("/chat/completions") for path in _PATHS)
