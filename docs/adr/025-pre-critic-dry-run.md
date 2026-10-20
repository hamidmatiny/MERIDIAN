# ADR-025: Pre-critic checks and degraded reviews

- Status: Accepted

## Context

A critic was spending a full round on patches that could not parse, and on repairs that rewrote rows outside the failing set. Structured-output failures were stored as `OTHER` rejections, which sent the patch back to Fix and counted toward the round cap.

## Decision

Before every critic call, MERIDIAN parses the remediation and the regression test, runs the sandbox static gate, and dry-runs a passing patch on a 200-row sample of the corrupted partition. The report records rows changed, rows changed outside the incident's failing set, contracts after, and PSI after. A syntax or gate failure returns to Fix, does not increment `critic_rounds`, and does not call the critic. The dry-run report is part of the critic prompt. The critic is told to cite it.

`OVER_BROAD_CHANGE` is an objection kind. It is kept only when the dry-run flag `over_broad` is true.

A critic structured-output failure stores `critic_degraded=True` with no objections. That is not a rejection. Two degraded critic calls set `needs_human`.

Sample rows included in any model prompt are capped at 10. On a later model turn, older tool payloads are replaced with their refs and a short summary.

## Consequences

- The critic sees a measurement it did not produce. It can still approve a broad change; the flag is evidence, and the objection is dropped when the flag is false.
- A syntax bounce consumes a Fix attempt, because Fix has to write the code again. It does not consume a critic round.
- Prompts are not rewritten to force a live pass. If a local model cannot fill the schema, the run reports that model.

## Alternatives considered

- Count a syntax failure as a critic rejection. Rejected because it spends critic tokens on a check `ast.parse` already answered.
- Treat a degraded critic call as a rejection. Rejected because a parse failure is not a judgment of the patch.
- Send every failing row to the model. Rejected because the prompt size dominated the call. The cap is 10.
