"""Run a gated remediation inside a Ray worker.

The worker is a separate process. It gets an empty environment and a temporary
working directory. See ADR-019 for what macOS does and does not enforce.
"""

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from meridian.sandbox.gate import check_source

_ENTRY = {"repair", "quarantine", "check"}
_WARMED_GENERATION = -1


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandboxed call. ``frame`` is set only when ``kind`` is ok."""

    kind: str
    detail: str
    frame: pd.DataFrame | None
    rows_out: int
    rows_in: int


def execute(
    source: str,
    frame: pd.DataFrame,
    *,
    entry: str,
    timeout_s: float,
    memory_mb: int,
    num_cpus: int,
    allow_pandera: bool = False,
) -> SandboxResult:
    """Gate ``source``, then run ``entry`` on ``frame`` in a Ray task."""
    if entry not in _ENTRY:
        return SandboxResult("runtime", f"unknown entry {entry}", None, 0, len(frame))
    reasons = check_source(source, allow_pandera=allow_pandera)
    if reasons:
        return SandboxResult("rejected", "; ".join(reasons), None, 0, len(frame))
    payload = frame.to_parquet(index=False)
    result = _remote_call(
        source,
        payload,
        entry,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        num_cpus=num_cpus,
        allow_pandera=allow_pandera,
        rows_in=len(frame),
    )
    from meridian.tracing import note_sandbox

    note_sandbox(entry, result.kind, result.rows_in, result.rows_out)
    return result


def _remote_call(
    source: str,
    payload: bytes,
    entry: str,
    *,
    timeout_s: float,
    memory_mb: int,
    num_cpus: int,
    allow_pandera: bool,
    rows_in: int,
) -> SandboxResult:
    import ray
    from ray.exceptions import GetTimeoutError, RayTaskError

    memory_bytes = max(memory_mb, 1) * 1024 * 1024
    _ensure_ray(num_cpus)
    pending = (
        ray.remote(_worker)
        .options(memory=memory_bytes)
        .remote(source, payload, entry, allow_pandera, memory_bytes)
    )
    try:
        blob, rows_out, total = ray.get(pending, timeout=timeout_s)
    except GetTimeoutError:
        ray.cancel(pending, force=True)
        return SandboxResult("timeout", "sandbox timed out", None, 0, rows_in)
    except RayTaskError as exc:
        return SandboxResult("runtime", str(exc), None, 0, rows_in)
    frame = pd.read_parquet(io.BytesIO(bytes(blob)))
    return SandboxResult("ok", "", frame, int(rows_out), int(total))


def _worker(
    source: str,
    payload: bytes,
    entry: str,
    allow_pandera: bool,
    memory_bytes: int,
) -> tuple[bytes, int, int]:
    """Ray entry. Clears the environment, runs the source, returns parquet bytes.

    The address-space cap covers only the untrusted source. Parquet decode and encode run
    outside it: pyarrow's thread pool reserves address space that the cap would refuse.
    """
    import os
    import tempfile

    cwd = str(Path.cwd())
    saved = dict(os.environ)
    frame = pd.read_parquet(io.BytesIO(payload))
    rows_in = len(frame)
    previous = _limit_memory(memory_bytes)
    try:
        os.environ.clear()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            namespace: dict[str, object] = {"__builtins__": _builtins(allow_pandera)}
            exec(source, namespace, namespace)  # trusted harness; the source was gated
            fn = namespace.get(entry)
            if not callable(fn):
                raise RuntimeError(f"missing {entry}")
            if entry == "check":
                fn(frame)
                out = frame
            elif entry == "quarantine":
                out = _apply_mask(frame, fn(frame))
            else:
                out = fn(frame)
            _restore_limit(previous)
            previous = None
            if not isinstance(out, pd.DataFrame):
                raise RuntimeError("remediation did not return a dataframe")
            blob = out.to_parquet(index=False)
            return bytes(blob), len(out), rows_in
    except (SchemaError, SchemaErrors) as exc:
        raise RuntimeError(f"schema check failed: {exc}") from None
    finally:
        _restore_limit(previous)
        os.environ.clear()
        os.environ.update(saved)
        os.chdir(cwd)


def _warm_worker() -> None:
    """Touch the worker runtime once. The remediation timeout does not cover this."""


def _ensure_ray(num_cpus: int) -> None:
    """Join the process cluster. A new cluster is warmed once; a shared one is reused."""
    import os

    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    import ray

    from meridian.agents.ray_runtime import start_ray

    global _WARMED_GENERATION
    generation = start_ray(num_cpus)
    if generation == _WARMED_GENERATION:
        return
    ray.get(ray.remote(_warm_worker).remote())
    _WARMED_GENERATION = generation


def _limit_memory(memory_bytes: int) -> tuple[int, int] | None:
    """Ask the OS for an address-space cap. macOS does not honor a small one."""
    return _apply_limit(sys.platform, memory_bytes)


def _apply_limit(platform: str, memory_bytes: int) -> tuple[int, int] | None:
    """Allow ``memory_bytes`` of new address space above what the worker already maps.

    Only the soft limit moves, so the worker can restore it after the task. Returns the
    limits to restore, or None when no cap was set.
    """
    import os

    if platform == "darwin" or os.environ.get("MERIDIAN_SANDBOX_RLIMIT", "1") == "0":
        return None
    mapped = _mapped_bytes()
    if mapped is None:
        return None
    import resource

    try:
        previous = resource.getrlimit(resource.RLIMIT_AS)
        hard = previous[1]
        soft = mapped + memory_bytes
        if hard != resource.RLIM_INFINITY:
            soft = min(soft, hard)
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    except (ValueError, OSError):
        return None
    return (int(previous[0]), int(previous[1]))


def _restore_limit(previous: tuple[int, int] | None) -> None:
    if previous is None:
        return
    import resource

    try:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    except (ValueError, OSError):
        return


def _mapped_bytes() -> int | None:
    """Address space this process already maps, from ``/proc``. None off Linux."""
    import os

    try:
        with Path("/proc/self/statm").open(encoding="ascii") as handle:
            pages = int(handle.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return pages * os.sysconf("SC_PAGE_SIZE")


def _apply_mask(frame: pd.DataFrame, mask: object) -> pd.DataFrame:
    if isinstance(mask, pd.Series):
        flags = mask.astype(bool)
    elif isinstance(mask, (list, tuple)):
        if len(mask) != len(frame):
            raise RuntimeError("quarantine mask length does not match the frame")
        flags = pd.Series(list(mask), index=frame.index).astype(bool)
    else:
        raise RuntimeError("quarantine mask must be a series or a list")
    if len(flags) != len(frame):
        raise RuntimeError("quarantine mask length does not match the frame")
    return frame.loc[~flags.to_numpy()].copy()


def _builtins(allow_pandera: bool) -> dict[str, object]:
    allowed = {"pandas", "numpy", "math", "datetime"}
    if allow_pandera:
        allowed.add("pandera")

    def _import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        root = name.split(".", maxsplit=1)[0]
        if level != 0 or root not in allowed:
            raise ImportError(name)
        return __import__(name, globals_, locals_, fromlist, level)

    return {
        "__import__": _import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "print": print,
    }
