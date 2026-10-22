# ADR-031: The wall budget starts at process start and is enforced by a watchdog

## Context

`MERIDIAN_RUN_WALL_BUDGET_S` (900) was checked only when a graph node began, and the clock
started when the graph was compiled. Probes, model load, Ray init, and MCP startup sat
outside it. A node that blocked inside a model call was not checked again until the call
returned. A live unit-mismatch run produced no output for 16 minutes and never aborted as
`WALL_BUDGET`.

## Decision

The CLI starts a monotonic clock before probes, model load, Ray, MCP, and the graph. SIGALRM
fires when the budget is spent, which interrupts a blocked main thread. While the graph is
inside a node, the alarm is disarmed and `call_bounded` polls the same clock from the node
thread, so a provider that never returns is abandoned. The abandoned work runs on a daemon
thread and does not keep the process alive.

On abort the run is checkpointed when the graph has started, `needs_human` is set, and
`--slack` posts an Escalation with no approval buttons and without `interrupt()`. The process
exits non-zero. A wall abort is not a Slack wait.

The CLI and slack_app log one line per node start and end (`run_id`, `case`, `node`,
`attempt`, `elapsed_s`) and a heartbeat every 30 seconds while a model call is in flight.
Those lines go to stderr. stdout remains the final state JSON.

## Consequences

A blocked probe fails in 900 seconds instead of hanging until a human kills it. The graph's
own entry check still aborts a zero budget before triage, after Sentinel has opened the
cases. The signal handler is installed only for the CLI process and is restored when the run
ends. slack_app does not arm the alarm: it is a long-lived process, and a resume uses the
clock of that graph build only for the node poll.

## Alternatives

Polling only between model attempts leaves a stuck socket running. A watchdog thread that
kills worker threads cannot interrupt `Event.wait` or `recv`. Failing the run without a
checkpoint or a Slack escalation would hide the abort from the person who has to take the
incident.
