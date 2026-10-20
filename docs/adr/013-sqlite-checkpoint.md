# ADR-013: SQLite checkpointing for resumable runs

- Status: Accepted

## Context

Sentinel can finish and the process can die before Triage. Re-running Sentinel would re-read every partition and could assign a new `detected_at`. The graph needs a checkpoint that resumes at the next node.

## Decision

The compiled graph uses `langgraph-checkpoint-sqlite`. The file path is `MERIDIAN_CHECKPOINT_PATH` (default `data/meridian-checkpoints.sqlite`). The thread id is the run id. State is `MeridianState`: Sentinel writes `incidents`, and the conditional edge sends an empty list to END and a non-empty list to Triage. Triage writes one `TriageResult` per incident.

A test interrupts after Sentinel, invokes again with the same thread id, and checks that Sentinel's function ran once and Triage then completes.

## Consequences

- Resume does not re-enter a node that already finished, so Sentinel does not start Ray again on that path.
- The checkpoint file is local state. `data/` is gitignored.
- The checkpointer is SQLite, not Postgres. The audit database remains a later phase.
- `detected_at` is captured when the graph is built, so the incidents stored in the checkpoint keep that timestamp.

## Alternatives considered

- An in-memory checkpointer. Rejected because a killed process would lose the Sentinel result.
- Postgres now. Rejected. The audit log is Phase 7, and SQLite is enough to prove resume on one machine.
- Re-running the whole graph after a crash. Rejected because Sentinel is the expensive step and its incidents are already decided.
