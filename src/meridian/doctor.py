"""Report process and model health. This command never kills a process."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from meridian.config import Settings, get_settings
from meridian.llm.resident import budget_problem, read_ollama_sizes, size_report


@dataclass(frozen=True)
class Report:
    """One snapshot the doctor prints."""

    ollama: str
    omniroute: str
    bind: str
    ray_sessions: tuple[str, ...]
    omniroute_pids: tuple[str, ...]
    memory: str
    loaded: str
    models: str
    downsize: str


def main() -> int:
    """Print the snapshot and the commands a human can run. Exit 0."""
    print(render(collect()))
    return 0


def collect(settings: Settings | None = None) -> Report:
    """Read local services. Network and process calls stay inside the helpers."""
    chosen = get_settings() if settings is None else settings
    sizes = read_ollama_sizes()
    problem = budget_problem(chosen, sizes)
    return Report(
        ollama=_ollama(chosen),
        omniroute=_omniroute(chosen),
        bind=_bind(),
        ray_sessions=tuple(_ray_sessions()),
        omniroute_pids=tuple(_omniroute_pids()),
        memory=_memory(),
        loaded=_loaded(),
        models=size_report(chosen, sizes),
        downsize="" if problem is None else problem,
    )


def render(report: Report) -> str:
    """The human-readable snapshot, including commands and no kill."""
    lines = [
        f"ollama {report.ollama}",
        f"omniroute {report.omniroute}",
        f"omniroute_bind {report.bind}",
        f"memory {report.memory}",
        f"loaded_models {report.loaded}",
        report.models,
    ]
    if report.downsize:
        lines.append(report.downsize)
    if report.ray_sessions:
        lines.append("orphaned ray sessions older than 1h:")
        lines.extend(f"  {path}" for path in report.ray_sessions)
        lines.append("  command: uv run ray stop --force")
        lines.extend(f"  command: rm -rf {path}" for path in report.ray_sessions)
    else:
        lines.append("orphaned ray sessions older than 1h: none")
    if len(report.omniroute_pids) > 1:
        lines.append("duplicate omniroute processes: " + " ".join(report.omniroute_pids))
        lines.extend(f"  command: kill {pid}" for pid in report.omniroute_pids)
    else:
        shown = report.omniroute_pids[0] if report.omniroute_pids else "none"
        lines.append(f"omniroute processes: {shown}")
    lines.append("doctor does not kill processes")
    return "\n".join(lines)


def _ollama(settings: Settings) -> str:
    status, body = _get(settings.ollama_base_url.rstrip("/") + "/api/version", None)
    if status != 200:
        return "unreachable"
    return "ok " + body


def _omniroute(settings: Settings) -> str:
    url = settings.omniroute_base_url.rstrip("/") + "/models"
    secret = settings.omniroute_api_key
    token = "" if secret is None else secret.get_secret_value().strip()
    status, _body = _get(url, token or None)
    if status == 0:
        return "unreachable"
    if status == 401:
        return "AUTH_FAILED"
    if status == 200:
        return "ok"
    return f"http {status}"


def _bind() -> str:
    completed = subprocess.run(
        ["lsof", "-nP", "-iTCP:20128", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if "127.0.0.1:20128" in line:
            return "127.0.0.1:20128"
        if "*:20128" in line or "0.0.0.0:20128" in line:
            return "*:20128"
    return "not listening"


def _ray_sessions(root: Path = Path("/tmp/ray")) -> list[str]:
    if not root.is_dir():
        return []
    cutoff = time.time() - 3600
    found: list[str] = []
    for path in sorted(root.glob("session_*")):
        try:
            if path.stat().st_mtime < cutoff:
                found.append(str(path))
        except OSError:
            continue
    return found


def _omniroute_pids() -> list[str]:
    completed = subprocess.run(
        ["ps", "-axo", "pid,command"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[str] = []
    for line in completed.stdout.splitlines():
        if "omniroute serve" not in line or "grep" in line:
            continue
        pid = line.split(maxsplit=1)[0]
        if pid.isdigit():
            pids.append(pid)
    return pids


def _memory() -> str:
    total = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
        check=False,
    )
    pages = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
    swap = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        capture_output=True,
        text=True,
        check=False,
    )
    free = _free_pages(pages.stdout)
    memsize = total.stdout.strip() or "unknown"
    swap_text = swap.stdout.strip() or "unknown"
    return f"memsize_bytes={memsize} free_pages={free} swap={swap_text}"


def _loaded() -> str:
    try:
        completed = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "ollama command not found"
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return "none"
    return " | ".join(lines[1:])


def _free_pages(text: str) -> str:
    for line in text.splitlines():
        if "Pages free" in line:
            return line.split(":")[-1].strip()
    return "unknown"


def _get(url: str, token: str | None) -> tuple[int, str]:
    request = Request(url)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, response.read().decode()
    except HTTPError as exc:
        return exc.code, ""
    except (URLError, TimeoutError, OSError):
        return 0, ""


if __name__ == "__main__":
    raise SystemExit(main())
