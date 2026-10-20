# ADR-005: Deterministic Verifier (no LLM judgment for pass/fail)

- Status: Accepted

## Context

A critic can approve a patch that does not actually repair the failing partition. A model asked "did this pass?" can also disagree with the data contract. The question MERIDIAN needs answered is narrower: after the patch, do the failing rows still fail the contract?

## Decision

Pass/fail is a `VerificationResult` produced by replaying the failing data against the patched contract. The planned implementation runs that replay on Ray. No LLM call is part of the verdict. `passed` is true only when `failures_remaining` is zero; the model rejects a result that claims both.

Phase 0 defines the result record and that invariant. It does not run Ray or load a dataset.

## Consequences

- A green critic and a red verifier can disagree. The verifier wins, and the graph routes back to Fix until `fix_attempts` hits its cap.
- Replay cost and duration are measured (`rows_checked`, `duration_ms`) instead of inferred from a model score.
- The verifier cannot talk itself into a pass. A bug in the replay is a code bug, which is the kind of failure a test can pin down.
- Empty successes are allowed by the contract (`passed` true, zero failures, possibly zero rows). Whether an empty partition is a valid success is a later-phase policy, not a Phase 0 guess.

## Alternatives considered

- LLM-as-judge for pass/fail. Rejected because the contract already defines the check, and a model judgment is not repeatable.
- Diff review without replaying data. Rejected because a plausible diff can leave the original rows failing.
- Unit tests alone, with no data replay. Rejected as the pass/fail signal. The generated regression test rides along on the `Patch`; the verifier still has to replay the failing partition.
