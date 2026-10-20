# ADR-016: Deterministic grounding guardrail on LLM evidence

- Status: Accepted

## Context

A root-cause model can name an event or a column profile that no tool returned. That citation looks specific and is unscorable. The check has to be the same on every run, with no second model in the loop.

## Decision

After the model returns a hypothesis, a deterministic pass walks the refs collected from this case's tool payloads. A ref is any string stored under a JSON key named `ref`.

- An `Evidence.reference` that is not in that set is removed.
- `causal_event_id` is kept only when `event:<id>` was retrieved. Otherwise the id is cleared.
- Each removal counts as one grounding violation. Confidence drops by 0.25 per violation, and does not go below 0.
- The pass never inserts a reference and never substitutes a retrieved id for a missing one.
- `hypothesis.grounding_violations` is that count. A clean citation leaves it at 0.

An unparseable causal id is treated as an id no tool returned, so it is cleared and counted. The hypothesis id and incident id are assigned by the node. That corrects the parent link. It is not evidence.

If structured output does not parse, or the tool budget ends before a valid draft, the node still emits a hypothesis: empty evidence, no causal id, confidence 0, and zero violations. Nothing is invented to fill it.

## Consequences

- A hallucinated ref cannot survive into the case. The score then sees a missing cause instead of a fabricated one.
- The CLI fake cites the newest event on the incident's dataset, using only `id`, `ts`, and `dataset` from tool JSON. It does not read the causal-event marker, an injected-fault label, or the log file. Scripted models under `tests/` may still key off that marker.
- The guardrail can only check refs the tools actually stamped. A true statement with no ref is dropped.
- Lowering confidence is fixed arithmetic, so two runs with the same tool output and the same draft get the same violations.

## Alternatives considered

- Asking the model to self-check its citations. Rejected. That check moves when the model does.
- Filling a missing causal id with the event the tools returned. Rejected. That would launder a guess into a grounded claim.
- Rejecting the whole hypothesis on one bad ref. Rejected. The other citations can still be real, and the violation count is what a later benchmark can read.
