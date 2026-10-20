# ADR-002: Typed Pydantic contracts as the only inter-agent interface

- Status: Accepted

## Context

Each node in the graph consumes the previous node's output. If that output is a raw dict, a typo in a field name becomes a runtime surprise in a later node, and a reviewer cannot see the interface without reading prompts. The same records also need to be stored, sent to Slack, and checked into the repo as JSON Schema.

## Decision

Every value that crosses a node boundary is a frozen Pydantic v2 model in `src/meridian/contracts/`. One module holds one concept. Models use strict types, reject unknown fields, and document each field. Timestamps are timezone-aware UTC. Parent ids are checked when `MeridianState` holds both the parent and the child.

`scripts/export_schemas.py` writes the JSON Schema of every contract to `schemas/`. A test fails if those files drift from the models.

Nodes must not pass raw dicts to each other. Dicts are allowed only inside a contract field that is explicitly a document, such as a sampled failure row.

## Consequences

- A bad payload fails at the model boundary with a validation error, before the next node runs.
- The JSON Schema files are a reviewable artifact. Changing a contract updates the schema in the same change.
- Strict mode does not coerce strings into UUIDs or datetimes in Python. JSON parsing still accepts the encoded forms, which is what the round-trip tests cover.
- `Hypothesis` and `Patch` carry their own ids even though the first sketch of those records only listed parent ids. Child records (`hypothesis_id`, `patch_id`) need a parent id to match.

## Alternatives considered

- Raw dicts or untyped JSON between nodes. Rejected because the project rule forbids them and they hide breakage until a later node.
- `TypedDict` only. Rejected because it does not enforce the UTC, range, and cross-id rules at runtime.
- Protobuf or another IDL as the source of truth. Rejected for Phase 0. Pydantic is already the runtime type system; JSON Schema is generated from it rather than maintained by hand.
