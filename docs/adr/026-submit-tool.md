# ADR-026: Submit the contract as a tool

- Status: Accepted

## Context

ADR-022 sends structured output with `method="json_schema"`. On Ollama 0.40.6 that becomes a `format` field on `POST /api/chat`. For `qwen3.5:4b` and `gemma4:e4b` the server answers HTTP 501 `structured output is unavailable` for a JSON schema and for `format: "json"`. The OpenAI-compatible `response_format` path returns the same 501. `langchain-ollama` `with_structured_output(method="json_schema")` sends that `format` value, so the call fails before the model speaks. A tool call on the same endpoint returns HTTP 200. Phase 4.2 classified the 501 as a transient error, retried it, and then swallowed the failure, so triage and the critic degraded with no tokens.

## Decision

Every structured node finishes by calling a tool whose arguments are the Pydantic contract: `submit_triage_assessment`, `submit_hypothesis`, `submit_patch`, and `submit_verdict`.

- Root cause and fix bind their read-only MCP tools plus the submit tool. The model may submit on any turn.
- Triage and the critic bind only their submit tool.
- On the last turn of the tool budget, the node binds only the submit tool and adds the instruction `submit now`.
- Tool arguments are validated with Pydantic. A `ValidationError` is sent back as the tool result and retried at most twice, then the node degrades as it does today.
- A plain-text reply gets one corrective retry, then the node degrades.
- `json_schema` remains available, and `json_mode` remains the third choice. A node uses either one only when the capability probe says that provider and model accept it.

At the start of a live run the probe checks `tool_calling`, `structured_json_schema`, and `structured_json_mode` for every hop. Results are cached in `data/capabilities.json`, keyed by provider, model, and Ollama version. The choice is submit-tool, then JSON schema, then JSON mode. If a node and its whole fallback chain have none of the three, the run aborts before any agent work.

HTTP 501, or a provider message that a capability is unavailable, is `UNSUPPORTED`. That hops to the next provider and sets `fallback_used`. It is not a silent degrade. If the chain is exhausted, the run aborts. `OMNIROUTE_BASE_URL` is read only from settings.

A pre-critic dry-run runs for `DATA_REPAIR` and `QUARANTINE`. `ESCALATE` and `RERUN_UPSTREAM` record `dry_run=None` after the static checks pass.

Each run reports `hypotheses_from_budget_fallback` and `structured_success_rate` per node (successful submits divided by submit attempts).

## Consequences

- Local qwen and gemma runs no longer depend on Ollama's `format` channel.
- A provider that accepts neither tools nor JSON fails the run in the open, before Sentinel spends a model call.
- Fakes and tests that do not set an output method stay on `json_schema`, so the existing scripted models keep working.
- The capability cache can remember a miss. Deleting `data/capabilities.json` probes again.

## Alternatives considered

- Keep `json_schema` and treat 501 as a retry. Rejected. The same request 501s every time, and the failure was being swallowed.
- Use `format: "json"` when schema mode 501s. Rejected. Ollama 0.40.6 returns 501 for that value too.
- Parse free text locally. Rejected. ADR-022 already refused a local extractor, and the tool channel is accepted by the server.
- Number this decision ADR-023. Rejected. ADR-023 through ADR-025 already record the free-tier router, the budget, and the pre-critic dry-run.
