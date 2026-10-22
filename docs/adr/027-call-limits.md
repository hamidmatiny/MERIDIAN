# ADR-027: Bound every model call and the whole run

## Context

Local thinking models were called with a 32768-token context and no output cap. One critic
call ran for hours and reported about 260000 output tokens. A graph run has no wall clock, so
a stuck call holds the GPU until a human kills it. OmniRoute was also probed without a bearer
token, so an HTTP 401 looked like a model that simply had no structured method.

## Decision

Each call waits at most `MERIDIAN_LLM_TIMEOUT_S` (180). That limit is a deadline around the
whole invoke, including a stream. An httpx read timeout resets on every token, so a thinking
model can stream for many minutes and still record `timed_out=false`. A 789 s fix call did
that. The deadline is `Future.result(timeout=...)`. Output is capped at
`MERIDIAN_LLM_MAX_OUTPUT_TOKENS` (2048), which Ollama receives as `num_predict`. Local context
is `MERIDIAN_LLM_NUM_CTX` (8192). A timeout is transient: one retry, then the next provider.
Other transient HTTP errors still retry twice. Each timeout is an `LLMCall` with `timed_out`
set. Ollama `load_duration` is stored as `load_duration_ms`, and the run report sums it.

Submit-tool nodes send `think=false` (`ChatOllama(reasoning=False)`). `LLMCall` records
`think` and Ollama `eval_count`. Qwen3.5 and Gemma 4 think by default on Ollama 0.40.6, and
those thinking tokens count toward `num_predict`.

`keep_alive` for the fix and critic model tags is the run wall budget, so both can stay
resident while the run lasts. The server is also started with `OLLAMA_MAX_LOADED_MODELS=2`
and `OLLAMA_KEEP_ALIVE=15m`. A run that exceeds `MERIDIAN_RUN_WALL_BUDGET_S` (900) aborts as
`WALL_BUDGET`, checkpoints that state, and sets `needs_human`.

A live start refuses when the distinct configured local tags exceed `MERIDIAN_MAX_MODEL_GB`
(26) and names the largest tag to downsize. A tag used by several nodes is counted once.
Triage, root cause, and fix share `qwen3.5:27b-q4_K_M` (17 GB). The critic is `gemma4:12b`
(7.7 GB). That pair is 24.7 GB. `qwen3.5:4b` stays on disk and is not in the configured set.
Adding it back as a third tag would be 28.7 GB and the gate would refuse the start. The cap
is not raised to hide that.

OmniRoute calls send `OMNIROUTE_API_KEY` as a bearer token. The OpenAI client
rejects an empty key and would otherwise read `OPENAI_API_KEY`, so a blank
setting is sent as the placeholder `missing`. The capability probe sends no
header when the setting is blank. A probe that receives only HTTP 401
is `AUTH_FAILED`. That row is not cached. A connection error stays `UNREACHABLE` and is not
cached either. The key is never logged.

`python -m meridian.doctor` prints reachability, auth, the bind address, Ray sessions older
than one hour, duplicate OmniRoute processes, memory, and loaded models. It prints the
commands a human can run. It does not kill anything. The graph runner and the live tests call
`ray.shutdown()` in a `finally` block. Unit tests time out at 60 seconds and live tests at
1200.

## Consequences

Sharing the 27B tag across triage, root cause, and fix keeps the distinct total at 24.7 GB,
under the cap. A third tag that pushes the distinct total over 26 GB still refuses the start.
Timeouts and wall-budget aborts show up in the run report. An OmniRoute 401 is distinct from
a model that answered and has no structured method.

## Alternatives

- Leave context at 32768 and rely on a human to kill a stuck call. Rejected: one call already
  ran for hours.
- Cache `AUTH_FAILED` for 24 hours. Rejected: the cache key does not include the credential,
  so a fixed key would stay rejected.
- Count only the fix and critic models toward the 26 GB cap. Rejected: the check is the
  configured set, and triage is part of that set.
