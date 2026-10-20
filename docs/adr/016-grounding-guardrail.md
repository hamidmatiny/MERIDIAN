# ADR-016: Deterministic grounding guardrail on LLM evidence

- Status: Accepted

## Context

A root-cause model can name an event or a column profile that no tool returned. That citation looks specific and is unscorable. The check has to be the same on every run, with no second model in the loop.

## Decision

After the model returns a hypothesis, a deterministic pass walks this case's tool payloads.

- A string stored under a JSON key named `ref` is a ref.
- `event:<uuid>` and `<uuid>` are the same event. Both forms are stored when either one appears.
- An event id seen anywhere in a tool payload is grounded. `list_events` returns those ids, so a later `get_event` is not required.
- An `Evidence.reference` that does not match that set is removed.
- `causal_event_id` is kept when its normalized event id was seen. Otherwise the id is cleared.
- Each removal counts as one grounding violation and is stored as a `GroundingViolation`: the field, the raw model value, the normalized value, and a short sample of the refs it was checked against. Confidence drops by 0.25 per violation, and does not go below 0.
- The pass never inserts a reference and never substitutes a retrieved id for a missing one.
- `hypothesis.grounding_violations` is that count. A clean citation leaves it at 0.

An unparseable causal id is treated as an id no tool returned, so it is cleared and counted. The hypothesis id and incident id are assigned by the node. That corrects the parent link. It is not evidence.

If structured output does not parse, or the tool budget ends before a valid draft, the node still emits a hypothesis: empty evidence, no causal id, confidence 0, and zero violations. Nothing is invented to fill it.

## Consequences

- A hallucinated ref cannot survive into the case. The bench score reports both `raw_causal_correct` (the id the model named, recovered from the violation when the guardrail cleared it) and `causal_event_correct` (the id that survived). A guardrail false negative is the case where the first is true and the second is false.
- Any UUID in a tool payload is treated as an event id. That matches `list_events`. It can also ground an id that happened to appear for another reason.
- The CLI fake cites the newest event on the incident's dataset, using only `id`, `ts`, and `dataset` from tool JSON. It does not read the causal-event marker, an injected-fault label, or the log file. Scripted models under `tests/` may still key off that marker.
- The guardrail can only check refs the tools actually stamped. A true statement with no ref is dropped.
- Lowering confidence is fixed arithmetic, so two runs with the same tool output and the same draft get the same violations.

## Alternatives considered

- Asking the model to self-check its citations. Rejected. That check moves when the model does.
- Filling a missing causal id with the event the tools returned. Rejected. That would launder a guess into a grounded claim.
- Rejecting the whole hypothesis on one bad ref. Rejected. The other citations can still be real, and the violation count is what a later benchmark can read.
