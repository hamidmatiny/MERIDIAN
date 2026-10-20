# ADR-020: Cross-provider critic inputs and objection taxonomy

- Status: Accepted

## Context

ADR-003 assigns the review to Grok so it does not share the author's provider. The critic still needs a fixed input set and a fixed objection vocabulary. A free-text rejection cannot be counted, and a review that can see the injected fault is grading against the label rather than the evidence.

## Decision

The critic sees the incident, the hypothesis including its evidence, the remediation kind, the diff, and the regression test. It does not see the injected fault, the clean frame, or the generator seed.

`Verdict.objections` is a list of `Objection` values. The kind is one of:

- `WRONG_ROOT_CAUSE`
- `UNSAFE_CODE`
- `DATA_LOSS_RISK`
- `WRONG_REMEDIATION_KIND`
- `MISSING_TEST`
- `OTHER`

An approval has an empty list. A rejection with no parsed objection is recorded as `OTHER`. The case stores `critic_rejection_rate`, the share of verdicts on that case that rejected the patch.

`--live` calls `require_anthropic()` first, then checks `MERIDIAN_TRIAGE_MODEL`, `MERIDIAN_ROOTCAUSE_MODEL`, `MERIDIAN_FIX_MODEL`, and `MERIDIAN_CRITIC_MODEL`, then calls `require_xai()`. The critic model is `MERIDIAN_CRITIC_MODEL` via `langchain-xai`. The fix model is `MERIDIAN_FIX_MODEL` via `langchain-anthropic`.

## Consequences

- Rejection rate is a per-case number, not a benchmark claim.
- A structured-output failure is a rejection of kind `OTHER` and counts as a round. The loop still stops at three rounds.
- The critic cannot "check its work" against the label. That comparison lives in the bench scorer (ADR-021).
- A live run now needs both provider keys. A missing xAI key is reported only after the Anthropic key and the model names are present, so an unset Anthropic key does not mention xAI.

## Alternatives considered

- Free-text objections. Rejected because the retry prompt and the rejection rate need a stable kind.
- Showing the critic the injected fault. Rejected because the review would stop being a stand-in for production, where no label exists.
- A same-provider critic with a different prompt. Already rejected in ADR-003.
