# ADR-017: Explicit bounded tool-loop subgraph

- Status: Accepted

## Context

Root cause has to call tools and then stop. A prebuilt ReAct agent hides the stop condition inside a library loop, and it starts the next case before this run can order the cases by the severity triage just assigned. Severity is unknown until triage returns.

## Decision

The graph is `START → sentinel → triage → dispatch → root_cause → END`. Sentinel still sends one triage branch per open case. `dispatch` is deferred, so it runs once after every triage branch has merged. It logs the cases most-severe first (`SEV1`, then `SEV2`, `SEV3`, `SEV4`) and sends one root-cause branch each. The sends stay parallel. The log is not a barrier between those branches.

The tool loop is its own `StateGraph`, compiled inside the root-cause node. It is not `create_react_agent`. One plan step either requests one tool or asks for a structured hypothesis. The cap is `MERIDIAN_ROOTCAUSE_MAX_TOOL_CALLS`, default 8. When the count of executed tools reaches the cap, the loop does not call another tool. It forces one structured answer and then runs the grounding pass from ADR-016.

The model name is `MERIDIAN_ROOTCAUSE_MODEL`. The prompt is `src/meridian/prompts/root_cause.v1.md`, recorded as `prompt_version` on each `LLMCall`. Every tool call is a `ToolCall` on the run: node, tool, args hash, ref count, latency, and whether the payload parsed. Nodes return only the new calls. `append_tool_calls` concatenates them, the same way `append_llm_calls` does.

The checkpoint is unchanged. The thread id is still the run id. An interrupt after Sentinel still does not re-run Sentinel.

## Consequences

- The cap is an edge in a graph we can read, not a parameter of a prebuilt agent.
- Ordering is visible in the log and does not serialize the root-cause branches.
- A case that exhausts the budget still leaves a valid `Hypothesis` on the case.
- `dispatch` is a join, not an agent. It returns no model calls.

## Alternatives considered

- `create_react_agent` with a recursion limit. Rejected. The limit counts messages, not tool calls, and the forced final answer is harder to keep inside the contract.
- Sorting in the Sentinel fan-out. Rejected. Severity does not exist until triage has finished.
- A plain edge from triage to root cause. Rejected. Each branch would start root cause before the other severities were known, so the log could not show one order for the run.
