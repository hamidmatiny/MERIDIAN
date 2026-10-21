"""Run a gated remediation inside a Ray worker.

The worker is a separate process. It gets an empty environment and a temporary
working directory. See ADR-019 for what macOS does and does not enforce.
"""

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    return _remote_call(
        source,
        payload,
        entry,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        num_cpus=num_cpus,
        allow_pandera=allow_pandera,
        rows_in=len(frame),
    )


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
    """Ray entry. Clears the environment, runs the source, returns parquet bytes."""
    import os
    import tempfile

    _limit_memory(memory_bytes)
    cwd = str(Path.cwd())
    saved = dict(os.environ)
    try:
        os.environ.clear()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            namespace: dict[str, object] = {"__builtins__": _builtins(allow_pandera)}
            exec(source, namespace, namespace)  # trusted harness; the source was gated
            fn = namespace.get(entry)
            if not callable(fn):
                raise RuntimeError(f"missing {entry}")
            frame = pd.read_parquet(io.BytesIO(payload))
            rows_in = len(frame)
            if entry == "check":
                fn(frame)
                out = frame
            elif entry == "quarantine":
                out = _apply_mask(frame, fn(frame))
            else:
                out = fn(frame)
            if not isinstance(out, pd.DataFrame):
                raise RuntimeError("remediation did not return a dataframe")
            blob = out.to_parquet(index=False)
            return bytes(blob), len(out), rows_in
    finally:
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


def _limit_memory(memory_bytes: int) -> None:
    """Ask the OS for an address-space cap. macOS does not honor a small one."""
    _apply_limit(sys.platform, memory_bytes)


def _apply_limit(platform: str, memory_bytes: int) -> None:
    if platform == "darwin":
        return
    import resource

    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        return


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
