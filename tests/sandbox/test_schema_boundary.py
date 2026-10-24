"""A pandera failure leaves the Ray worker as RuntimeError, which pickles cleanly."""

import sys
from datetime import UTC, date, datetime
from typing import ClassVar

import pandas as pd
import pytest
from pandera.errors import SchemaError

from meridian.agents import sentinel
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.partition_ref import PartitionRef
from meridian.sandbox.execute import _apply_limit, _restore_limit, _worker

_SOURCE = """
def check(frame):
    import pandera.pandas as pa
    schema = pa.DataFrameSchema({"speed": pa.Column(str)})
    schema.validate(frame)
"""


_EXECUTE = sys.modules["meridian.sandbox.execute"]


class _Resource:
    RLIMIT_AS = 9
    RLIM_INFINITY = -1
    hard = -1
    calls: ClassVar[list[tuple[int, tuple[int, int]]]] = []

    @classmethod
    def getrlimit(cls, kind: int) -> tuple[int, int]:
        del kind
        return (cls.hard, cls.hard)

    @classmethod
    def setrlimit(cls, kind: int, limits: tuple[int, int]) -> None:
        cls.calls.append((kind, limits))


def test_linux_rlimit_follows_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _Resource.calls = []
    _Resource.hard = -1
    monkeypatch.setitem(sys.modules, "resource", _Resource)
    monkeypatch.setattr(_EXECUTE, "_mapped_bytes", lambda: 1000)
    monkeypatch.setenv("MERIDIAN_SANDBOX_RLIMIT", "0")
    assert _apply_limit("linux", 2048) is None
    assert _Resource.calls == []
    monkeypatch.setenv("MERIDIAN_SANDBOX_RLIMIT", "1")
    previous = _apply_limit("linux", 2048)
    assert _Resource.calls == [(9, (3048, -1))]
    assert previous == (-1, -1)
    _restore_limit(previous)
    assert _Resource.calls[-1] == (9, (-1, -1))


def test_linux_rlimit_stays_under_a_finite_hard_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _Resource.calls = []
    _Resource.hard = 2500
    monkeypatch.setitem(sys.modules, "resource", _Resource)
    monkeypatch.setattr(_EXECUTE, "_mapped_bytes", lambda: 1000)
    monkeypatch.setenv("MERIDIAN_SANDBOX_RLIMIT", "1")
    assert _apply_limit("linux", 2048) == (2500, 2500)
    assert _Resource.calls == [(9, (2500, 2500))]


def test_no_proc_means_no_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _Resource.calls = []
    monkeypatch.setitem(sys.modules, "resource", _Resource)
    monkeypatch.setattr(_EXECUTE, "_mapped_bytes", lambda: None)
    monkeypatch.setenv("MERIDIAN_SANDBOX_RLIMIT", "1")
    assert _apply_limit("linux", 2048) is None
    assert _Resource.calls == []
    _restore_limit(None)
    assert _Resource.calls == []


def test_worker_schema_error_is_runtime() -> None:
    frame = pd.DataFrame({"speed": [1.0]})
    payload = frame.to_parquet(index=False)
    with pytest.raises(RuntimeError, match="schema check failed"):
        _worker(_SOURCE, payload, "check", True, 0)


def test_sentinel_schema_error_is_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = PartitionRef(
        dataset="vehicle_telemetry",
        partition_date=date(2026, 10, 24),
        parquet_path="x.parquet",
        metadata=PartitionMetadata(
            dataset="vehicle_telemetry",
            partition_date=date(2026, 10, 24),
            arrived_at=datetime(2026, 10, 24, tzinfo=UTC),
        ),
        reference_profile_path="profile.json",
    )

    def boom(found: PartitionRef, moment: datetime) -> list[object]:
        del found, moment
        raise SchemaError(None, None, "bad column")

    monkeypatch.setattr(sentinel, "evaluate_partition", boom)
    with pytest.raises(RuntimeError, match="schema check failed"):
        sentinel._remote_entry(ref.model_dump_json(), "2026-10-24T00:00:00+00:00")
