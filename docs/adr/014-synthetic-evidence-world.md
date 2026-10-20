# ADR-014: Synthetic evidence world with causal and distractor events

- Status: Accepted

## Context

Root cause has to be scored against a known cause. The injected frame shows the fault, but it does not say which change produced it. A model that invents a plausible story cannot be checked.

## Decision

Every injection writes `events.jsonl` next to the partition. Each line is a `ChangeEvent`: id, timestamp, kind, dataset, author, summary, and a short diff excerpt. Kinds are `SCHEMA_CHANGE`, `DEPLOY`, `CONFIG_CHANGE`, `UPSTREAM_DELAY`, `BACKFILL`, and `DEPENDENCY_BUMP`.

Exactly one event is causal. Its id is `uuid5` of the injection inputs and is stored on `InjectedFault.causal_event_id`. That id is not part of `InjectedFault.id`, so existing fault ids stay stable. `require_causal_event` refuses a label whose id is missing or repeated in the file. The mapping is:

| Fault | Causal kind |
| --- | --- |
| `SCHEMA_DRIFT` | `SCHEMA_CHANGE` renaming the column |
| `UNIT_MISMATCH` | `CONFIG_CHANGE` switching the unit |
| `LATE_PARTITION` | `UPSTREAM_DELAY` |
| `NULL_EXPLOSION` | `BACKFILL` writing nulls |
| `DUPLICATE_KEYS` | `DEPLOY` repeating keys |
| `DISTRIBUTION_SHIFT` | `DEPENDENCY_BUMP` moving the distribution |

The other three mappings were chosen for this phase so every fault has one kind. Distractors may use the same kind.

The default log adds 8 distractors from the seed. Index 0 is on the same dataset, three days earlier. Index 1 is on a different dataset, 30 minutes after the causal event, inside a 2 hour window. The rest follow a formula of seed and index, not a shared random stream, so they do not move when the frame generator consumes the seed. A clean partition gets the same distractors and no causal event.

## Consequences

- A hypothesis is right when `causal_event_id` matches the label and the suspect columns overlap the affected columns. `score` in `src/meridian/bench/score_root_cause.py` computes those two values. It is a unit-tested stub for Phase 6, not a published accuracy.
- The event log is deterministic for a dataset, fault, intensity, and seed.
- The writer checks the file. The model itself cannot see a path, so the check is not a field validator.

## Alternatives considered

- Hiding the cause only in the corrupted frame. Rejected because many faults look alike and the score would be a judgment call.
- One unique kind per fault, with no distractor of that kind. Rejected. The agent should have to read the event, not just the kind.
- Drawing distractors from `default_rng(seed)` after the frame injector. Rejected because the two streams would be coupled and easy to shift by accident.
