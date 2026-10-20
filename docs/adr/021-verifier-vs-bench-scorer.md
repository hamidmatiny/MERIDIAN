# ADR-021: Production verifier versus the bench scorer

- Status: Accepted

## Context

The verifier has to be valid in production, where the clean frame and the injected fault do not exist. The benchmark still wants to know whether a repair landed on the original rows. Those are different questions. Putting the clean frame into the verifier would make a passing replay depend on a label the production path will never have.

## Decision

The verifier answers only questions it can answer from the corrupted partition, the contracts, the reference profile, and the patch.

- `DATA_REPAIR`: the sandboxed repair must leave zero incidents from `quality.evaluate`, the regression test must pass, rows dropped must be at most `MERIDIAN_MAX_ROW_LOSS` (default 1%), and PSI on the profiled columns must be back under the stored threshold.
- `QUARANTINE`: the remaining rows must pass the contracts, the regression test must pass, and the quarantined fraction must be at most `MERIDIAN_MAX_QUARANTINE` (default 0.25, so an injection of about 10% nulls can pass).
- `RERUN_UPSTREAM`: regenerate the partition with `generate_dataset` from `PartitionRef.generation_seed` and `row_count`, and score it with an on-time arrival. The regenerated frame must pass the contracts. The verifier does not compare it to a clean frame it does not have.
- `ESCALATE`: not executed. `needs_human` is set.

`VerificationResult` carries `remediation_kind` and `checks`, one pass/fail entry per check. `passed` is true only when every check passed and `failures_remaining` is 0.

`src/meridian/bench/score_fix.py` is the only place that receives the clean frame. It returns `kind_appropriate`, `repaired_matches_clean` (numeric tolerance), and `rows_lost_pct` in percentage points (1.0 means one percent of the rows). Unit tests call it. The README does not publish a score from it.

## Consequences

- A verifier pass does not mean the repaired frame equals the clean frame. It means the contracts, the row-loss cap, and PSI hold.
- A bench score can disagree with the verifier. That disagreement is the point of keeping them apart.
- `generation_seed` and `row_count` are stored on the partition ref by the CLI. A partition that lacks them fails the rerun check instead of guessing a seed.
- The verifier does not import the injector and does not read `InjectedFault`.

## Alternatives considered

- Passing the clean frame into the verifier behind a flag. Rejected because a flag that changes the meaning of `passed` will eventually be left on.
- Treating PSI as a bench-only metric. Rejected because a unit conversion that leaves the profiled range is visible in production without a label, and the threshold is already the contract.
- Scoring row loss only in the benchmark. Rejected because a repair that drops the partition is a production failure even when the remaining rows validate.
