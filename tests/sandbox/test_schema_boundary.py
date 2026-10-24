"""A pandera failure leaves the Ray worker as RuntimeError, which pickles cleanly."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from pandera.errors import SchemaError

from meridian.agents import sentinel
from meridian.contracts.partition_metadata import PartitionMetadata
from meridian.contracts.partition_ref import PartitionRef
from meridian.sandbox.execute import _apply_limit, _worker

_SOURCE = """
def check(frame):
    import pandera.pandas as pa
    schema = pa.DataFrameSchema({"speed": pa.Column(str)})
    schema.validate(frame)
"""


def test_linux_rlimit_follows_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []

    class _Resource:
        RLIMIT_AS = 9

        @staticmethod
        def setrlimit(kind: int, limits: tuple[int, int]) -> None:
            calls.append((kind, limits))

    monkeypatch.setitem(__import__("sys").modules, "resource", _Resource)
    monkeypatch.setenv("MERIDIAN_SANDBOX_RLIMIT", "0")
    _apply_limit("linux", 2048)
    assert calls == []
    monkeypatch.setenv("MERIDIAN_SANDBOX_RLIMIT", "1")
    _apply_limit("linux", 2048)
    assert calls == [(9, (2048, 2048))]


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
