# ADR-012: Versioned prompt files and per-call LLM accounting

- Status: Accepted

## Context

A triage prompt that lives only in a Python string is hard to review, and a later cost report cannot reconstruct what was sent or how many tokens came back. Phase 7 will price model calls. The rows have to exist before that work starts.

## Decision

Prompts are markdown files named `<name>.<version>.md` under `src/meridian/prompts/`. Triage loads `triage.v1.md` through `load_prompt` and stores `prompt_version` on `TriageResult`. The model name is `MERIDIAN_TRIAGE_MODEL` from settings. No model id is hard-coded.

Every attempt, including a failed parse, appends an `LLMCall`: node, model, prompt version, input tokens, output tokens, latency, attempt number, and whether structured output validated. Token counts are read from the response usage metadata. The default CLI uses `FakeTriageModel`, which returns a fixed SEV4 assessment and fixed token counts. `--live` is the path that constructs `ChatAnthropic`.

## Consequences

- A prompt change is a new file version, and old runs still name the version they used.
- `MeridianState.llm_calls` is the ledger Phase 7 will sum. This phase does not compute a dollar cost.
- The fake model is deterministic, so unit tests and the default CLI do not need a network or an API key.
- `--live` calls `Settings.require_llm()` first and exits with the missing key names when they are absent.

## Alternatives considered

- Embedding the prompt in the triage function. Rejected because a review of the wording would be mixed into the control flow.
- Logging token counts only for successful calls. Rejected because a retry still spends tokens, and the ledger would under-count.
- A default live model id in code. Rejected. The name comes from configuration.
