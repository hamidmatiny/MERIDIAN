# ADR-032: Postmortem times come from recorded events, and Slack is audited

## Context

Phase 5 postmortems built the timeline by adding one second per row to `detected_at`,
then overwriting the human row with `decided_at` when it was later. Those offsets were
not the node times. `time_to_resolution_s` was detection until the decision, which mixed
the machine run and the human wait. An ESCALATE patch still uploaded its `diff`, which
for the unit-mismatch escalation was a two-line file with no remediation. Approver ids
were printed as raw text, and Slack replies, uploads, and button clicks were not in an
append-only log.

ADR-031 is the process wall watchdog. This record is the next number.

## Decision

Each node start and end is appended, when it happens, to `data/progress/<run_id>.jsonl`
with the UTC time of that call. The postmortem timeline uses those timestamps. A case
with no events is marked `times not recorded (pre-5.2 run)` and is not given synthetic
offsets. `machine_time_s` is detection until the `slack_approval` start. `human_wait_s`
is the first recorded post time until `decided_at`. `time_to_resolution_s` is the sum.
The two waits are stored together or not at all. A later upsert of an open case keeps
the original `posted_at`.

`data/audit/slack.jsonl` is append-only. One JSON line per mention, reply, post, upload,
button, decision, duplicate, unauthorized click, and button removal. Upload lines store
the byte count and sha256 of the exact diff. Token-shaped strings are redacted and
secret field names are dropped. ESCALATE and RERUN_UPSTREAM upload no diff. Approvers
render as `<@USERID>` in Slack mrkdwn and in the postmortem.

## Consequences

Pre-5.2 checkpoints cannot grow a timeline after the fact. Their reports say the times
were not recorded. A resume does not move `posted_at`, so human wait stays the original
post. The audit file lives under `data/` and is not committed. A second `--final` bench
lock is unrelated and is recorded in the bench ADR.

## Alternatives

Keep synthesizing offsets so every report has a dense timeline. Rejected: the times
would still be false. Store the events only in the LangGraph checkpoint. Rejected: the
approval node interrupts before it returns, so the start time would be lost.
