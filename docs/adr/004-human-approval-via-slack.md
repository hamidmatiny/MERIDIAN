# ADR-004: Human approval via Slack is mandatory before merge

- Status: Accepted

## Context

A verified patch is still a code change to a data pipeline. MERIDIAN must not merge that change on its own. The person on call needs the diff, the evidence, and a clear way to approve, reject, or let the request time out. A missing response must not be treated as approval.

## Decision

No patch merges without an `ApprovalDecision`. The decision is one of `APPROVED`, `REJECTED`, or `TIMEOUT`. The planned gate uses a LangGraph `interrupt()` and a Slack Bolt message that carries the diff and the evidence. The human replies in the configured approval channel. `TIMEOUT` is a distinct outcome so a silent channel escalates instead of merging.

Phase 0 defines `ApprovalDecision` and `Settings.require_slack()`. It does not open a Slack connection. The bot token, signing secret, and approval channel are optional at import and required only when the gate runs.

## Consequences

- The merge path has a human in it. Verifier success is not sufficient.
- Slack credentials live in the environment, never in the repo. `.env.example` lists the names with placeholder values.
- `TIMEOUT` has to be handled by a later phase. It is not an implicit approve and not an implicit reject.
- The approval record stores who decided, when (UTC), and a comment, including an empty comment.

## Alternatives considered

- Auto-merge when the verifier passes. Rejected because a green replay is not a license to change production pipelines unattended.
- Email or a ticketing system as the only gate. Rejected for the first version. Slack is where the on-call conversation already happens. A second channel can be added later without changing the `ApprovalDecision` contract.
- GitHub pull-request review as the only gate. Rejected as the primary control because the graph needs an interrupt it can wait on, with a timeout. A pull request can still be the merge mechanism underneath the Slack decision.
