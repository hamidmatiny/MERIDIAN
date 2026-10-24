"""Append-only audit tables. The application inserts. Verify detects a rewrite."""

from sqlalchemy import JSON, BigInteger, Column, Integer, MetaData, String, Table
from sqlalchemy.engine import Connection, Engine

TABLES = ("audit_events", "llm_calls", "tool_calls", "runs")
metadata = MetaData()


def _table(name: str) -> Table:
    return Table(
        name,
        metadata,
        Column(
            "id",
            BigInteger().with_variant(Integer, "sqlite"),
            primary_key=True,
            autoincrement=True,
        ),
        Column("prev_hash", String(64), nullable=False),
        Column("row_hash", String(64), nullable=False),
        Column("payload", JSON, nullable=False),
    )


AUDIT_EVENTS = _table("audit_events")
LLM_CALLS = _table("llm_calls")
TOOL_CALLS = _table("tool_calls")
RUNS = _table("runs")
BY_NAME = {table.name: table for table in (AUDIT_EVENTS, LLM_CALLS, TOOL_CALLS, RUNS)}


def create_all(bind: Engine | Connection) -> None:
    """Create the four tables when they are missing."""
    metadata.create_all(bind)
