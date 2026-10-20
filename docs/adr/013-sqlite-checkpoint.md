# ADR-013: SQLite checkpointing for resumable runs

- Status: Accepted

## Context

Sentinel can finish and the process can die before Triage. Re-running Sentinel would re-read every partition and could assign a new `detected_at`. The graph needs a checkpoint that resumes at the next node.

## Decision

The compiled graph uses `langgraph-checkpoint-sqlite`. The file path is `MERIDIAN_CHECKPOINT_PATH` (default `data/meridian-checkpoints.sqlite`). The thread id is `MeridianState.run_id`.

Sentinel writes one `IncidentCase` per incident into `cases`. The conditional edge returns a LangGraph `Send` for each case that has no triage yet, or END when `cases` is empty. Each triage branch receives only its own case and returns that case plus the calls it made. `merge_cases` keeps every incident id and replaces only the key being written. `append_llm_calls` concatenates the new calls onto the run. A node must return only the calls it just made, because the reducer appends.

A test interrupts after Sentinel, invokes again with the same thread id, and checks that Sentinel's function ran once and each open case is then triaged.

## Consequences

- Resume does not re-enter a node that already finished, so Sentinel does not start Ray again on that path.
- Parallel triage branches reduce by incident id, so one branch cannot replace another incident's case or its calls.
- The checkpoint file is local state. `data/` is gitignored.
- The checkpointer is SQLite, not Postgres. The audit database remains a later phase.
- `detected_at` is captured when the graph is built, so the incidents stored in the checkpoint keep that timestamp.

## Alternatives considered

- An in-memory checkpointer. Rejected because a killed process would lose the Sentinel result.
- Postgres now. Rejected. The audit log is Phase 7, and SQLite is enough to prove resume on one machine.
- Re-running the whole graph after a crash. Rejected because Sentinel is the expensive step and its incidents are already decided.
