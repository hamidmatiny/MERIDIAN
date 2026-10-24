# ADR-037: One-command demo and Compose

## Context

A reviewer may have no Slack workspace, no Ollama, and no API key. They still need to see the graph, the approval gate, an applied patch, and a trace.

## Decision

`python -m meridian.demo` injects `UNIT_MISMATCH` and `NULL_EXPLOSION`, runs the graph on the heuristic models, and asks approve or reject on the terminal. A non-interactive stdin approves, so CI and Compose cannot hang. The first line says the models are fake and the run demonstrates the pipeline, not model quality. Compose starts Postgres, Jaeger, and that demo. Live Ollama stays on the host at `host.docker.internal:11434` and is not containerized. The demo database password is the local Compose credential.

## Consequences

The demo finishes without network access to a model. Jaeger at `localhost:16686` shows the demo trace. The demo does not measure model quality, and its output says so.

## Alternatives

Replaying a recorded bench run was rejected because it would skip the approval interrupt. Containerizing Ollama was rejected because the macOS app does not run inside this Compose file.
