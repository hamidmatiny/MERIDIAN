# ADR-022: Native JSON-schema structured output

- Status: Accepted

## Context

Triage, the root-cause final answer, the fix agent, and the critic all request a Pydantic object through `with_structured_output`. The LangChain default method is forced tool calling. On `claude-sonnet-5-5` the API rejects a forced tool choice. The same path warns, and then does not force the tool, when extended thinking is enabled. Live triage on `claude-haiku-5-5` and live root cause on `claude-sonnet-5-5` both came back with `succeeded` false: the model returned prose, the parser had no tool call, and triage left `llm_severity` unset. Disabling thinking would only cover the thinking case. It would not cover Sonnet 5.5, which rejects forced tool choice whether or not thinking is on.

## Decision

Structured-output nodes call `bind_structured`, which passes `method="json_schema"` and `include_raw=True`.

- For Claude, that is Anthropic's structured-output `output_config`, not a tool choice. `langchain-anthropic` implements it as `method="json_schema"`.
- For Grok, `langchain-xai` implements the same method as xAI structured outputs.
- Tool-planning turns in root cause and fix still use `bind_tools`. Only the schema-shaped answer uses JSON schema. Thinking is left at the provider default for those planning turns.

A parsed object is accepted as the Pydantic model or as a dict that validates as that model. A parse failure is still a failed `LLMCall` and, for triage, a degraded result on the severity floor.

## Consequences

- The live triage test requires `succeeded` true and `llm_severity` set. A provider that rejects JSON schema fails that test in the open.
- Prompts are unchanged. The fix is the output channel, not the wording.
- A future model that supports neither JSON schema nor forced tools needs a new decision. This one does not add a second parser per node.

## Alternatives considered

- Disabling extended thinking on the structured-output clients. Rejected. Sonnet 5.5 still rejects forced tool choice with thinking off, and the tool-planning turns can keep thinking.
- Retrying the forced tool call with a stronger prompt. Rejected. The API does not accept the call shape, so a prompt cannot make it succeed.
- Parsing free text with a local JSON extractor. Rejected. The provider already has a schema channel, and a local extractor would drift from it.
