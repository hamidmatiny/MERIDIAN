# ADR-024: Budget caps and the provider circuit breaker

- Status: Accepted

## Context

Paid calls need a price before they run, and a response that omits usage cannot be charged honestly. Anthropic has returned `usage_metadata` of 0/0 while the real counts sat on `response_metadata["usage"]`. A credit error used to be swallowed, after which the critic kept running.

## Decision

Prices live in `src/meridian/llm/prices.yaml` as USD per one million input and output tokens. A paid model with no row is refused. Nothing in the code guesses a price. Before a paid call, the estimated input tokens times the input price is checked against the remaining run cap and the remaining day cap. After the call, the provider's real usage is stored in `data/spend.sqlite`. A paid response with no usage aborts the run.

Provider errors are classified as billing (400 with credit text, or 402), auth (401, 403), quota (429 with quota or limit text), transient (5xx, timeouts, other 429), or bad output. Billing, auth, and quota set `MeridianState.aborted`, trip a run-wide gate, checkpoint through the normal graph end, and make the CLI exit 1. No fallback hypothesis is written and no later node calls a model. Transient errors retry twice with backoff, then the next hop in the chain is used.

`python -m meridian.spend` prints today's paid spend and the all-time paid spend per provider.

## Consequences

- The committed price table is empty until a published rate is added. Paid models therefore refuse until that row exists.
- Free calls record zero USD and do not write a spend row.
- A fatal error stops every case, not only the case that saw it.

## Alternatives considered

- Hard-code public list prices. Rejected because a guessed or stale number would bill against the wrong cap.
- Fall back to a free model after a credit error. Rejected because the operator asked the run to stop, not to continue on a different provider.
- Keep going when usage is missing and record zero. Rejected because that hides spend.
