# ADR-035: OpenTelemetry spans, off unless an exporter is configured

## Context

A reviewer needs to see one run as a tree: the run, each case, each node, each model call, each tool call, and each sandbox execution. Unit tests must not pay for a tracer, and a span must not store a prompt or a secret.

## Decision

Spans are created only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set or `MERIDIAN_TRACE=console`. Otherwise the helpers return a no-op and do not construct a provider. OTLP goes to a local Jaeger. Attributes are the model name, provider, prompt version, token counts, latency, timeout, think flag, attempt, tool name, evidence ref count, and the case outcome. Prompt text and secret-like attribute names are dropped.

## Consequences

A default test run exports nothing. A demo with the endpoint set shows the tree in Jaeger at `localhost:16686`. The prompt body is not recoverable from a trace.

## Alternatives

Always-on console export was rejected because it would print a span for every node in the suite. A custom logger was rejected because Jaeger already renders the tree.
