# ADR-018: Remediation kinds

- Status: Accepted

## Context

Not every data fault can be undone by a row transform. A null cannot be reconstructed from the row that lost it. A late partition was generated correctly and arrived late; rewriting its rows does not address the cause. A patch that always emits `repair(df) -> df` would invent values or hide a case that should go to a person.

## Decision

`Patch.remediation_kind` is one of four values.

- `DATA_REPAIR`: a `repair(df) -> df` function. Used when a reversible change explains the fault, such as a unit conversion, a renamed column, or duplicated keys.
- `QUARANTINE`: a `quarantine(df)` predicate. True marks a row to drop. Used when the bad values cannot be reconstructed and the dropped fraction stays within `MERIDIAN_MAX_QUARANTINE` (default 0.25).
- `RERUN_UPSTREAM`: no row transform. The verifier regenerates the partition from the pipeline seed with the lateness reverted.
- `ESCALATE`: no automated fix. The case is marked `needs_human`.

The fix prompt tells the model it may choose any of the four. The diff adds `remediations/<incident_id>.py`. `regression_test` is a pandera-based `check(df)` for that module. One patch id is derived from the incident id and the prompt version, so a later verdict still points at the patch after a retry.

## Consequences

- A null explosion is not required to come back as a repair. Quarantine, escalation, and rerun are valid. The causal event for `NULL_EXPLOSION` is a `BACKFILL`, so asking the pipeline to rerun the partition is a valid response to that event, alongside quarantine and escalation. `kind_appropriate` uses that set. The other faults are unchanged.
- A late partition is a rerun, not a dataframe edit.
- Escalation still passes through the critic and the verifier. The verifier does not execute the module. It records a failed check named `escalated` and sets `needs_human`, so the loop does not retry.
- The default kind on an older `Patch` is `ESCALATE`, so a patch that never chose a kind does not silently transform rows.

## Alternatives considered

- Always emit a repair function. Rejected because it forces the model to invent values for faults that are not invertible.
- A free-text remediation field. Rejected because the verifier has to choose an execution path, and a closed set of kinds is that choice.
- Dropping escalation from the critic loop. Rejected because an independent review can still object that a case was escalated too early, or that a repair should have been an escalation.
