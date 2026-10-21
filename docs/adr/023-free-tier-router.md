# ADR-023: Free-tier model router

- Status: Accepted

## Context

Live runs were bound to Claude and Grok. A credit error was recorded as a failed call with zero tokens, and the graph kept calling the other provider. Benchmarks also need a named model. An OmniRoute `auto` or combo id would make a run impossible to repeat.

## Decision

Each of triage, root cause, fix, and critic has `MERIDIAN_<NODE>_PROVIDER` and `MERIDIAN_<NODE>_MODEL`. Providers are `ollama`, `omniroute`, `anthropic`, `xai`, and `fake`. The default chain is Ollama, then OmniRoute. Paid providers are never in that default. OmniRoute rejects `auto` and combo names.

Ollama uses `ChatOllama` and `OLLAMA_BASE_URL`. OmniRoute uses `ChatOpenAI` against `OMNIROUTE_BASE_URL`. `OMNIROUTE_BASE_URL` comes only from this setting. Structured output uses `json_schema` (ADR-022) when the capability probe says the model accepts it, and a submit tool otherwise (ADR-026). Every `LLMCall` records `provider`, `model`, and `fallback_used`.

The pinned free models are `qwen3.5:4b` for triage, root cause, and fix, and `gemma4:e4b` for the critic.

## Consequences

- A machine without those Ollama tags cannot complete a live run. The schema is not loosened to compensate.
- `latest` is not the default. A moving tag would break the reproducibility rule that also rejects `auto`.
- Adding a paid hop requires the allow flag, a USD cap, and a row in the price table.

## Alternatives considered

- Keep Claude and Grok as the only live path. Rejected because a credit error should not be the default way to run.
- Accept OmniRoute `auto`. Rejected because a benchmark must name one model.
- Pin `qwen3.5:latest` and `gemma4:latest`. Rejected because `latest` changes under the same name.
