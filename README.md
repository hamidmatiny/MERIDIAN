# MERIDIAN

MERIDIAN is being built to detect data-quality failures, explain them, propose a fix, verify that fix against the failing data, and ask a human in Slack before anything merges. The path is a fixed LangGraph state machine. The goal is to shorten the time from incident to a verified, approved fix.

This repository is the Phase 0 foundation: typed contracts, environment settings, architecture decisions, JSON Schema export, and CI. The agents, the data replay, and the Slack gate are planned. Nothing here calls a model, Ray, or Slack.

## Target agent graph

The diagram is the target architecture. The nodes are not implemented.

```mermaid
flowchart TD
  sentinel[Sentinel]
  triage[Triage]
  rootCause[RootCause]
  fix[Fix]
  critic[Critic]
  verifier[Verifier]
  slack[SlackApproval]
  postmortem[Postmortem]

  sentinel --> triage --> rootCause --> fix --> critic --> verifier --> slack --> postmortem
  critic -->|"objections, at most 3 rounds"| fix
  verifier -->|"replay failed, at most 2 reroutes"| fix
```

| Node | Planned role |
| --- | --- |
| Sentinel | Run pandera contracts over partitions on Ray and emit an `Incident`. |
| Triage | Claude assigns severity and blast radius. |
| RootCause | Claude, with MCP tools, writes a `Hypothesis` and evidence. |
| Fix | Claude writes a `Patch` and a regression test. |
| Critic | Grok reviews the patch and returns a `Verdict`. |
| Verifier | Ray replays the failing data. Pass/fail is deterministic. |
| SlackApproval | LangGraph interrupt. A human approves, rejects, or the request times out. |
| Postmortem | Write the incident report. |

Loop limits, already enforced on `MeridianState`: Fix and Critic stop after 3 rounds. A failed verification may send the work back to Fix at most twice (3 Fix executions total), then the run escalates to a human.

## Phases

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repository, ADRs, Pydantic contracts, CI | in progress |
| 1 | Pandera contracts and an incident injector | planned |
| 2 | Graph skeleton: Sentinel to Triage | planned |
| 3 | RootCause agent with MCP tools | planned |
| 4 | Fix, Critic, and Verifier loop on Ray | planned |
| 5 | Slack approval gate and Postmortem | planned |
| 6 | MERIDIAN-Bench harness and published results | planned |
| 7 | OpenTelemetry, cost tracking, Docker Compose demo | planned |

## What is in this tree

- `src/meridian/contracts/` — frozen Pydantic models for the records above.
- `src/meridian/config.py` — settings loaded from the environment and `.env`. Missing keys do not crash import. `Settings.require_llm()` and `Settings.require_slack()` raise when a caller actually needs those credentials.
- `schemas/` — JSON Schema generated from the contracts.
- `docs/adr/` — ADR-001 through ADR-006.
- `.github/workflows/ci.yml` — the checks below.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest --cov=src/meridian --cov-fail-under=95 -q
uv run python scripts/export_schemas.py && git diff --exit-code schemas/
```

Copy `.env.example` to `.env` only when a later phase needs credentials. Phase 0 tests do not call external services.

## License

Apache-2.0. Copyright 2026 MERIDIAN contributors.
