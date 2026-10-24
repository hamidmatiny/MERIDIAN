# ADR-038: Pandera failures do not cross Ray as SchemaError

## Context

`SchemaError` and `SchemaErrors` warn on unpickle: pickling them does not preserve state. The warning fires on the driver when a Ray worker raises one. The two warnings in the clean bench log were failed sandbox checks.

## Decision

The sandbox worker and the Sentinel remote entry catch `SchemaError` and `SchemaErrors` and raise `RuntimeError("schema check failed: ...")` from that point. Ray pickles the `RuntimeError`. The verifier already maps a worker failure to a runtime sandbox result. Pass and fail stay the same; the exception type that crosses the process boundary changes.

## Consequences

A failed schema check no longer emits the pandera pickle warning. The failure text still names the schema problem. Callers that matched on the pandera class name outside the worker will not see that class.

## Alternatives

Filtering the warning was rejected because the warning is the symptom of a lossy pickle, not noise. Changing pandera was rejected because the boundary is ours.
