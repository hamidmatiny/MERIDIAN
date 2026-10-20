# ADR-003: Cross-family critic (free: Qwen generates, Gemma reviews)

- Status: Accepted

## Context

The Fix step will propose a code change and a regression test. A review by the same provider that wrote the patch can share the author's blind spots, especially when the review prompt is a paraphrase of the generation prompt. MERIDIAN's critic exists to catch that, inside a loop that is capped at three rounds.

## Decision

The critic must be a different model family from the fixer. `Settings` rejects a same-family pair at load. Qwen fix with Qwen critic is rejected. Qwen fix with Gemma critic is the free-tier pairing. Claude fix with Grok critic is the paid pairing, and it runs only when `MERIDIAN_ALLOW_PAID=1` and a USD budget cap is configured.

The free default chain is Ollama, then OmniRoute. Paid providers are not in that chain. The verdict still records `reviewer_model`, `approved`, `objections`, and the round (1 through 3). A structured-output failure sets `critic_degraded` and is not a rejection.

## Consequences

- A free run needs Ollama (and, on fallback, OmniRoute). It does not need `ANTHROPIC_API_KEY` or `XAI_API_KEY`. Those keys are required only for a paid hop, after `MERIDIAN_ALLOW_PAID=1` and a USD cap.
- A same-family pair fails at configuration load, before any model is called.
- A billing, auth, or quota error aborts the run. A transient error retries twice and then uses the next hop. See ADR-024.
- The verdict is data. It does not merge code and it does not decide that a replay passed. Those are separate steps.

## Alternatives considered

- Same-model self-critique. Rejected because it does not give an independent review.
- A second Claude call with a different prompt. Rejected for the same reason: one provider still supplies both judgments.
- Human review as the only critique. Rejected because the Slack gate is an additional control, not a substitute for a technical review before the human is interrupted.
- A vote across several models. Rejected because the loop cap is three rounds and the extra calls are not justified yet.
