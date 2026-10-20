# ADR-001: Deterministic LangGraph state machine over free-form multi-agent chat

- Status: Accepted

## Context

MERIDIAN has one job: shorten the time from a data-quality incident to a verified, human-approved fix. An open-ended multi-agent chat can wander, skip the verifier, or retry forever. The response path is known in advance, including the two loops and the point where a human must take over.

## Decision

The system will be a LangGraph state machine with a fixed node sequence:

Sentinel → Triage → RootCause → Fix → Critic → Verifier → SlackApproval → Postmortem

Two loops are part of the graph, and both have hard caps stored on `MeridianState`.

`critic_rounds` counts completed Fix/Critic exchanges. AGENTS.md caps that loop at 3 rounds, so the counter allows 0 through 3. Zero means the critic has not run. `Verdict.round` is a different field: the 1-based round stamped on one verdict, also limited to 1 through 3. The state does not require `verdict.round == critic_rounds`. Each value is checked against its own limit.

`fix_attempts` counts Fix executions, not only the trips back from the verifier. AGENTS.md says a failed verification may route back to Fix at most 2 times. Those two reroutes sit on top of the original Fix, so the counter allows 0 through 3: attempt 1 is the first patch, attempts 2 and 3 are the two allowed reroutes. Zero means Fix has not run. A value of 4 is rejected. The next verifier failure after the third Fix execution escalates to a human.

Phase 0 ships the state model and the caps. The graph nodes are planned and are not implemented yet.

## Consequences

- Behavior is inspectable. A run is a sequence of typed state transitions, not a transcript.
- Loop exhaustion is a state, not an accident. Callers can see that the cap was hit.
- Adding a step means editing the graph. Novel tool use belongs inside a node (RootCause and its MCP tools), not as a free conversation between nodes.
- The state model is frozen. Each node returns a new state rather than mutating a shared object.

## Alternatives considered

- Free-form multi-agent chat (AutoGen-style or an unconstrained tool loop). Rejected because the path, the retry budget, and the human gate would be implicit.
- A single agent with every tool. Rejected because the critic must be a different provider from the author, and the verifier must not be a model judgment.
- An imperative script with no graph. Rejected because the loops, the interrupt for Slack approval, and the later tracing all want an explicit state machine.
