# ADR-036: Tamper-evident audit log, Postgres or JSONL

## Context

Slack events, decisions, applies, model calls, tool calls, and runs need an append-only record. A reviewer should be able to see that a line was edited. Postgres is not always available, and the existing Slack log is JSONL.

## Decision

Each row stores `prev_hash` and `row_hash = sha256(prev_hash + canonical JSON)`. The first hashed row uses 64 zeroes as `prev_hash`. Alembic revision `0001` creates `audit_events`, `llm_calls`, `tool_calls`, and `runs`. `python -m meridian.audit verify` walks each chain and names the first broken row. When `DATABASE_URL` is unset or the server does not accept a connection, the same chain is appended to JSONL. Lines written before the chain have no hash and are skipped. The application only inserts.

## Consequences

Editing a payload without recomputing the digest fails verify at that row. A refused database does not drop the event. Legacy Slack lines stay as they were written.

## Alternatives

A database trigger that forbids UPDATE was rejected because the tamper test has to rewrite one row and then observe the failure. A signature key was rejected because this repository has no key custody.
