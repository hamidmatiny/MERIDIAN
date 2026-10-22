# ADR-028: Socket Mode, and resume from the checkpoint

- Status: Accepted

## Context

The approval gate has to stop the graph without holding a worker thread open for up to a day. Slack has to deliver the click to a process that may have restarted after the message was posted. A public HTTP endpoint would need a reachable host. This demo runs on one machine.

## Decision

`python -m meridian.slack_app` runs a Bolt Socket Mode handler and the graph worker in one process. The app token opens the socket. The bot token posts and uploads.

The graph node calls LangGraph `interrupt` after posting. The checkpoint is the same SQLite file the CLI uses. The CLI with `--slack` returns when the node interrupts. The button handler resumes that thread with `Command(resume=ApprovalDecision)`. A process that starts after the interrupt reads the same checkpoint and the same message log, so it does not post a second request.

SIGINT and SIGTERM close the socket, any MCP session the process opened, and Ray.

## Consequences

- Nothing waits inside the graph while the person is away. The wait is the socket process plus the checkpoint.
- Restarting `slack_app` does not lose the decision, because the checkpoint and the message ts are on disk.
- The machine running `slack_app` has to stay up for the click to arrive. A stopped socket does not receive it until the process starts again.
- Code before `interrupt` runs again on resume. Posts and uploads are keyed so the second pass is a no-op.

## Alternatives considered

- A blocking `sleep` inside the node until the click. Rejected. It holds the worker and dies on restart.
- An HTTP Request URL for Slack. Rejected for this demo. It needs a public address. Socket Mode does not.
- Resuming by re-running the graph from Sentinel. Rejected. The person approved a specific patch, and the models would run again.
