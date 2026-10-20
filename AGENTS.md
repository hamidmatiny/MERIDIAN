# MERIDIAN — Autonomous Data-Incident Response System

## Mission
A multi-agent system that detects data-quality failures in pipelines, determines root cause,
generates and verifies a fix, and requires human approval in Slack before anything merges.
It measures its own accuracy with a built-in benchmark (MERIDIAN-Bench).
Fixed goal: minimize time from incident to verified, approved fix.
Fixed path: a deterministic LangGraph state machine — NOT an open-ended agent chat.

## Agent graph (target, built across phases)
Sentinel → Triage → RootCause → Fix → Critic → Verifier → SlackApproval → Postmortem
- Sentinel: pandera contracts run over partitions on Ray → emits Incident
- Triage (Claude): severity + blast radius from lineage metadata
- RootCause (Claude + MCP tools): logs, git history, upstream schemas → Hypothesis with evidence
- Fix (Claude): Patch + generated pandera regression test
- Critic (Grok via langchain-xai): independent cross-provider review of the Patch → Verdict
- Verifier: deterministic Ray replay of failing data against the patch (no LLM)
- SlackApproval: LangGraph interrupt(); Slack Bolt message with diff + evidence; human approves/rejects
- Postmortem: incident report to Slack + repo
Loop limits: Fix↔Critic max 3 rounds; Verifier failure routes back to Fix max 2 times, then escalates to human.

## Stack
Python 3.11, uv, langgraph, langchain-core, langchain-anthropic, langchain-xai, pydantic v2,
pandera, ray, python-dotenv, slack-bolt, mcp, opentelemetry, postgres (audit log), pytest, ruff, mypy --strict.

## Phases (build ONE at a time; never start the next until the human confirms)
0 Repo, ADRs, Pydantic contracts, CI
1 pandera contracts + incident injector (seeded faults on sample pipeline data)
2 Graph skeleton: Sentinel → Triage only
3 RootCause agent with MCP tools
4 Fix / Critic / Verifier loop on Ray
5 Slack approval gate + Postmortem
6 MERIDIAN-Bench harness + published results + hard mode: clean data with realistic noise/outliers/legal edge values; report FP rate separately
7 OpenTelemetry tracing, cost tracking, Docker Compose demo, README with benchmark tables

## Hard rules
- Every architectural decision gets an ADR in docs/adr/ (Context / Decision / Consequences / Alternatives).
- All inter-agent data is a typed Pydantic model from src/meridian/contracts/. No raw dicts between nodes.
- No secrets in code. Read config via python-dotenv; keep .env.example current.
- No real LLM or Slack calls in unit tests. Use fakes and fixtures.
- Never claim a step works. Give the exact command for the human to run, and wait for the real output.
- Never invent benchmark numbers. All metrics come from MERIDIAN-Bench runs.
- Schema drift gate: `uv run python scripts/export_schemas.py && test -z "$(git status --porcelain schemas/)"`. Untracked schema files fail it, same as modifications.
- ruff and mypy --strict must pass before any phase is declared done.