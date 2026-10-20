# ADR-011: LLM severity bounded below by a deterministic rule floor

- Status: Accepted

## Context

Triage asks Claude for a severity. A model can under-call a schema break that already reaches two downstream datasets. The benchmark needs a severity that cannot quietly drop below a rule the repo can review.

## Decision

`severity_rules.FLOOR_TABLE` maps `(fault_type, downstream count)` to a minimum `Severity`. Downstream counts of 2 and above share one cell. `SCHEMA_DRIFT` with two or more downstreams floors at SEV2. The model returns a `TriageAssessment`. The stored `TriageResult` keeps `llm_severity`, `floor_severity`, and `final_severity`. `final_severity` is the more severe of the two, where SEV1 outranks SEV4. `severity` equals `final_severity`.

Structured output is attempted at most twice. If both attempts fail validation, `triage_degraded` is set, `llm_severity` stays unset, and `final_severity` is the floor. The graph does not raise.

## Consequences

- The model can raise severity. It cannot lower it.
- A degraded triage is still a complete `TriageResult`, so the rest of the run can continue.
- Changing a floor is a table edit, not a prompt edit.
- The ordering is an explicit rank, not the alphabetical order of the enum values.

## Alternatives considered

- Letting the model set severity with no floor. Rejected because a low answer on a wide blast radius would be unscored.
- Hard-coding severity with no model. Rejected because triage is specified to use Claude, and the model is still allowed to raise the floor.
- Retrying until the model complies. Rejected. Two attempts is the budget; after that the floor is the answer.
