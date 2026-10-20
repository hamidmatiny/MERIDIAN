"""A clean repair runs. A timeout is cancelled. Rejected source never starts Ray."""

import io

import pandas as pd
import pytest

from meridian.sandbox.execute import _limit_memory, _worker, execute


def test_rejected_source_does_not_run() -> None:
    frame = pd.DataFrame({"speed_mps": [1.0]})
    result = execute(
        "import os\ndef repair(df):\n    return df\n",
        frame,
        entry="repair",
        timeout_s=5,
        memory_mb=128,
        num_cpus=1,
    )
    assert result.kind == "rejected"
    assert result.frame is None


def test_unknown_entry_is_a_runtime_failure() -> None:
    frame = pd.DataFrame({"speed_mps": [1.0]})
    result = execute(
        "def repair(df):\n    return df\n",
        frame,
        entry="missing",
        timeout_s=5,
        memory_mb=128,
        num_cpus=1,
    )
    assert result.kind == "runtime"
    assert result.frame is None


def test_clean_repair_runs() -> None:
    frame = pd.DataFrame({"speed_mps": [72.0, 36.0]})
    source = (
        "def repair(df):\n"
        "    out = df.copy()\n"
        "    out['speed_mps'] = out['speed_mps'] / 3.6\n"
        "    return out\n"
    )
    result = execute(
        source,
        frame,
        entry="repair",
        timeout_s=30,
        memory_mb=256,
        num_cpus=1,
    )
    assert result.kind == "ok"
    assert result.frame is not None
    assert result.frame["speed_mps"].iloc[0] == 20.0
    assert result.rows_in == 2
    assert result.rows_out == 2


def test_timeout_is_enforced() -> None:
    frame = pd.DataFrame({"speed_mps": [1.0]})
    source = "def repair(df):\n    while True:\n        pass\n"
    result = execute(
        source,
        frame,
        entry="repair",
        timeout_s=1,
        memory_mb=256,
        num_cpus=1,
    )
    assert result.kind == "timeout"
    assert result.frame is None


def test_worker_repairs_quarantines_and_checks() -> None:
    frame = pd.DataFrame({"speed_mps": [72.0, None]})
    payload = frame.to_parquet(index=False)
    repair = "def repair(df):\n    return df.dropna()\n"
    blob, rows_out, rows_in = _worker(repair, payload, "repair", False, 1024)
    repaired = pd.read_parquet(io.BytesIO(blob))
    assert rows_in == 2
    assert rows_out == 1
    assert repaired["speed_mps"].iloc[0] == 72.0
    quarantine = "def quarantine(df):\n    return [True, False]\n"
    blob, rows_out, _rows_in = _worker(quarantine, payload, "quarantine", False, 1024)
    kept = pd.read_parquet(io.BytesIO(blob))
    assert rows_out == 1
    assert kept["speed_mps"].iloc[0] != 72.0 or kept["speed_mps"].isna().iloc[0]
    check = "def check(df):\n    return None\n"
    blob, rows_out, _rows_in = _worker(check, payload, "check", False, 1024)
    assert rows_out == 2
    checked = pd.read_parquet(io.BytesIO(blob))
    assert len(checked) == 2


def test_worker_rejects_a_missing_entry_and_a_bad_return() -> None:
    payload = pd.DataFrame({"speed_mps": [1.0]}).to_parquet(index=False)
    with pytest.raises(RuntimeError, match="missing"):
        _worker("def repair(df):\n    return df\n", payload, "check", False, 1024)
    with pytest.raises(RuntimeError, match="dataframe"):
        _worker("def repair(df):\n    return 1\n", payload, "repair", False, 1024)


def test_timeout_and_task_error_are_failures(monkeypatch) -> None:
    import importlib

    import ray
    from ray.exceptions import GetTimeoutError, RayTaskError

    sandbox = importlib.import_module("meridian.sandbox.execute")
    monkeypatch.setattr(sandbox, "_ensure_ray", lambda _cpus: None)

    class _Remote:
        def options(self, **_kwargs: object) -> "_Remote":
            return self

        def remote(self, *_args: object) -> object:
            return object()

    monkeypatch.setattr(ray, "remote", lambda _fn: _Remote())
    monkeypatch.setattr(ray, "cancel", lambda *_args, **_kwargs: None)
    frame = pd.DataFrame({"speed_mps": [1.0]})
    source = "def repair(df):\n    return df\n"

    def _timeout(*_args: object, **_kwargs: object) -> object:
        raise GetTimeoutError()

    monkeypatch.setattr(ray, "get", _timeout)
    timed = execute(source, frame, entry="repair", timeout_s=1, memory_mb=1, num_cpus=1)
    assert timed.kind == "timeout"

    def _task_error(*_args: object, **_kwargs: object) -> object:
        raise RayTaskError("worker", "traceback", RuntimeError("worker failed"))

    monkeypatch.setattr(ray, "get", _task_error)
    failed = execute(source, frame, entry="repair", timeout_s=1, memory_mb=1, num_cpus=1)
    assert failed.kind == "runtime"


def test_mask_and_import_edges() -> None:
    from meridian.sandbox.execute import _apply_limit, _apply_mask, _builtins

    frame = pd.DataFrame({"speed_mps": [1.0, 2.0]})
    kept = _apply_mask(frame, pd.Series([True, False]))
    assert len(kept) == 1
    with pytest.raises(RuntimeError, match="series or a list"):
        _apply_mask(frame, object())
    with pytest.raises(RuntimeError, match="length"):
        _apply_mask(frame, [True])
    _apply_limit("linux", 8 * 1024 * 1024 * 1024)
    importer = _builtins(True)["__import__"]
    assert callable(importer)
    pandas = importer("pandas")
    assert pandas is pd
    with pytest.raises(ImportError):
        importer("os")
    with pytest.raises(ImportError):
        importer("pandas", level=1)


def test_memory_limit_skips_the_address_space_cap_on_macos(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    _limit_memory(1024)


def test_memory_limit_is_best_effort_off_macos(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    _limit_memory(8 * 1024 * 1024 * 1024)
