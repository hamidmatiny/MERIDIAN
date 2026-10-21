"""The doctor prints commands. It does not run them."""

import os
import time
from pathlib import Path
from urllib.error import HTTPError

from meridian.config import Settings
from meridian.doctor import (
    Report,
    _bind,
    _loaded,
    _memory,
    _ollama,
    _omniroute,
    _omniroute_pids,
    _ray_sessions,
    collect,
    main,
    render,
)
from meridian.llm.resident import read_ollama_sizes


def test_render_names_commands_and_does_not_kill() -> None:
    text = render(
        Report(
            ollama="ok",
            omniroute="AUTH_FAILED",
            bind="127.0.0.1:20128",
            ray_sessions=("/tmp/ray/session_old",),
            omniroute_pids=("11", "22"),
            memory="memsize_bytes=1 free_pages=2 swap=none",
            loaded="none",
            models="combined local models 28.7 GB (cap 26 GB): qwen 17.0 GB",
            downsize="combined local models 28.7 GB (cap 26 GB): qwen 17.0 GB. Downsize qwen.",
        )
    )
    assert "AUTH_FAILED" in text
    assert "127.0.0.1:20128" in text
    assert "uv run ray stop --force" in text
    assert "rm -rf /tmp/ray/session_old" in text
    assert "kill 11" in text
    assert "doctor does not kill processes" in text


def test_collect_reports_a_snapshot(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr("meridian.doctor._ollama", lambda _settings: "ok")  # type: ignore[attr-defined]
    monkeypatch.setattr("meridian.doctor._omniroute", lambda _settings: "ok")  # type: ignore[attr-defined]
    monkeypatch.setattr("meridian.doctor._bind", lambda: "127.0.0.1:20128")  # type: ignore[attr-defined]
    monkeypatch.setattr("meridian.doctor._ray_sessions", list)  # type: ignore[attr-defined]
    monkeypatch.setattr("meridian.doctor._omniroute_pids", lambda: ["9"])  # type: ignore[attr-defined]
    monkeypatch.setattr("meridian.doctor._memory", lambda: "memsize_bytes=1")  # type: ignore[attr-defined]
    monkeypatch.setattr("meridian.doctor._loaded", lambda: "none")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "meridian.doctor.read_ollama_sizes",
        lambda: {"qwen3.5:4b": 4.0, "gemma4:e4b": 9.5},
    )  # type: ignore[attr-defined]
    settings = Settings(
        _env_file=None,
        triage_model="qwen3.5:4b",
        rootcause_model="qwen3.5:4b",
        fix_model="qwen3.5:4b",
        critic_model="gemma4:e4b",
        spend_path=str(tmp_path / "spend.sqlite"),
        max_model_gb=26,
    )
    text = render(collect(settings))
    assert "omniroute processes: 9" in text
    assert "orphaned ray sessions older than 1h: none" in text
    assert "13.5 GB" in text


class _Done:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


class _Body:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Body":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_helpers_read_process_and_http_output(monkeypatch: object, tmp_path: Path) -> None:
    def run(command: list[str], **_kwargs: object) -> _Done:
        if command[0] == "lsof":
            return _Done("node 1 TCP 127.0.0.1:20128 (LISTEN)\n")
        if command[0] == "ps":
            return _Done("11 omniroute serve --port 20128\n12 grep omniroute\n")
        if command[0] == "sysctl" and command[-1] == "hw.memsize":
            return _Done("1024\n")
        if command[0] == "vm_stat":
            return _Done("Pages free: 10.\n")
        if command[0] == "sysctl":
            return _Done("total = 0\n")
        if command[:2] == ["ollama", "ps"]:
            return _Done("NAME ID\nqwen3.5:4b abc\n")
        if command[:2] == ["ollama", "list"]:
            return _Done("NAME ID SIZE MODIFIED\nqwen3.5:4b abc 4.0 GB now\n")
        return _Done("", 1)

    monkeypatch.setattr("meridian.doctor.subprocess.run", run)  # type: ignore[attr-defined]
    monkeypatch.setattr("meridian.llm.resident.subprocess.run", run)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "meridian.doctor.urlopen",
        lambda *_args, **_kwargs: _Body(200, b'{"version":"0.40.6"}'),
    )  # type: ignore[attr-defined]
    settings = Settings(_env_file=None, ollama_base_url="http://127.0.0.1:9")
    assert "ok" in _ollama(settings)
    assert _omniroute(settings) == "ok"
    assert _bind() == "127.0.0.1:20128"
    assert _omniroute_pids() == ["11"]
    assert "memsize_bytes=1024" in _memory()
    assert "qwen3.5:4b" in _loaded()
    assert read_ollama_sizes()["qwen3.5:4b"] == 4.0
    old = tmp_path / "session_old"
    old.mkdir()
    fresh = tmp_path / "session_new"
    fresh.mkdir()
    past = time.time() - 7200
    os.utime(old, (past, past))
    assert _ray_sessions(tmp_path) == [str(old)]


def test_auth_failure_and_a_closed_port_are_distinct(monkeypatch: object) -> None:
    def refuse(*_args: object, **_kwargs: object) -> _Body:
        raise HTTPError(
            "http://127.0.0.1:20128/v1/models",
            401,
            "unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,  # type: ignore[arg-type]
        )

    def down(*_args: object, **_kwargs: object) -> _Body:
        raise OSError("down")

    monkeypatch.setattr("meridian.doctor.urlopen", refuse)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "meridian.doctor.subprocess.run",
        lambda *_args, **_kwargs: _Done("node 1 TCP *:20128 (LISTEN)\n"),
    )  # type: ignore[attr-defined]
    settings = Settings(_env_file=None)
    assert _omniroute(settings) == "AUTH_FAILED"
    assert _bind() == "*:20128"
    monkeypatch.setattr("meridian.doctor.urlopen", down)  # type: ignore[attr-defined]
    assert _ollama(settings) == "unreachable"
    assert main() == 0
