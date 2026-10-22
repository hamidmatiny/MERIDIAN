# ADR-029: Approver allowlist and one decision per patch

- Status: Accepted

## Context

Anyone in the channel can see the patch. Only some of them may accept it. Slack retries a button click when the handler is slow, and a person can click twice. A second apply, or a second postmortem, would be a bug. An escalation is not an approval: the message has Acknowledge, not Approve. Reject has to carry a comment. A silence of `MERIDIAN_APPROVAL_TIMEOUT_S` (default 86400) is a timeout, and a timeout is not an approval.

## Decision

`MERIDIAN_SLACK_APPROVERS` is a comma-separated list of Slack user ids. A click from any other id gets an ephemeral "not authorized" and does not resume the graph. The bot user id comes from `auth.test`. Mention text is parsed by stripping `<@BOTID>`, including the `|display-name` form. The display name is not consulted.

The first claim of a `patch_id` wins. The claim is an exclusive file create. A duplicate click or a Slack retry gets an ephemeral "already decided" and does not resume. The original message is updated to the decision, the approver, and the time, and the buttons are removed.

Reject opens a modal. An empty comment is rejected by the modal and does not resume. Acknowledge is `ACKNOWLEDGED`. It is not `APPROVED`. Timeout is `TIMEOUT`, sets `needs_human`, and is not an approval. Only `APPROVED` on a patch whose verifier passed, and whose case did not already need a human, is applied.

## Consequences

- The allowlist is user ids, so a renamed display name does not grant or remove access.
- One patch produces one decision even when Slack delivers the payload twice.
- An unauthorized click leaves the checkpoint untouched, so the allowed person can still decide.
- `ACKNOWLEDGED` is a new outcome beside `APPROVED`, `REJECTED`, and `TIMEOUT`. It records that a person saw an escalation. It does not apply a remediation.

## Alternatives considered

- Letting every channel member decide. Rejected. The channel is where the diff is visible, not the list of people who may merge it.
- Deduplicating by Slack's trigger id only. Rejected. Two real clicks from the same person have different trigger ids and would still apply twice.
- Treating Acknowledge as Approve. Rejected. An escalation has no passing verifier, and the button label would be a lie.
