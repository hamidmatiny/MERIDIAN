# ADR-003: Cross-provider critic (Claude generates, Grok reviews)

- Status: Accepted

## Context

The Fix step will propose a code change and a regression test. A review by the same provider that wrote the patch can share the author's blind spots, especially when the review prompt is a paraphrase of the generation prompt. MERIDIAN's critic exists to catch that, inside a loop that is capped at three rounds.

## Decision

Claude, through `langchain-anthropic`, authors the `Patch`. Grok, through `langchain-xai`, reviews that patch and returns a `Verdict`. The verdict records `reviewer_model`, `approved`, `objections`, and the round number (1 through 3). The critic does not share the author's provider.

Phase 0 defines the `Patch` and `Verdict` contracts and the round cap. It does not call either provider.

## Consequences

- Two credentials are required before the critic loop can run: `ANTHROPIC_API_KEY` and `XAI_API_KEY`. `Settings.require_anthropic()` and `Settings.require_xai()` are the explicit checks. Importing the package does not require them. A node calls only the provider it uses.
- A provider outage on either side stops the loop. That is preferred to a silent same-model review.
- Each round costs two model calls. The cap of three rounds bounds that cost.
- The verdict is data. It does not merge code and it does not decide that a replay passed. Those are separate steps.

## Alternatives considered

- Same-model self-critique. Rejected because it does not give an independent review.
- A second Claude call with a different prompt. Rejected for the same reason: one provider still supplies both judgments.
- Human review as the only critique. Rejected because the Slack gate is an additional control, not a substitute for a technical review before the human is interrupted.
- A vote across several models. Rejected because the loop cap is three rounds and the extra calls are not justified yet.
