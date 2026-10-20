# MERIDIAN

MERIDIAN is being built to detect data-quality failures, explain them, propose a fix, verify that fix against the failing data, and ask a human in Slack before anything merges. The path is a fixed LangGraph state machine. The goal is to shorten the time from incident to a verified, approved fix.

This repository has the Phase 0 contracts, the Phase 1 data contracts and injector, and the Phase 2 Sentinel → Triage graph. Root cause, the fix loop, and the Slack gate are not built. Unit tests do not call a model, Ray's object store beyond the local Sentinel tasks, or Slack.

## Target agent graph

Sentinel and Triage are implemented. The other nodes are planned.

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

  classDef built fill:#d9ead3,stroke:#38761d
  classDef planned fill:#f3f3f3,stroke:#999
  class sentinel,triage built
  class rootCause,fix,critic,verifier,slack,postmortem planned
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

Loop limits, already enforced on each `IncidentCase`: Fix and Critic stop after 3 rounds. A failed verification may send the work back to Fix at most twice (3 Fix executions total), then the run escalates to a human.

## Phases

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repository, ADRs, Pydantic contracts, CI | done |
| 1 | Pandera contracts and an incident injector | done |
| 2 | Graph skeleton: Sentinel to Triage | in progress |
| 3 | RootCause agent with MCP tools | planned |
| 4 | Fix, Critic, and Verifier loop on Ray | planned |
| 5 | Slack approval gate and Postmortem | planned |
| 6 | MERIDIAN-Bench harness and published results | planned |
| 7 | OpenTelemetry, cost tracking, Docker Compose demo | planned |

## What is in this tree

- `src/meridian/contracts/` — frozen Pydantic models. A run is `MeridianState`: partitions, `llm_calls`, and one `IncidentCase` per incident.
- `src/meridian/pipeline/` — seeded generators for the three fleet datasets, plus `lineage.yaml`.
- `src/meridian/quality/` — pandera models, PSI, the freshness SLA, and `evaluate`, which returns `Incident` records.
- `src/meridian/agents/` — Sentinel on Ray, the severity floor, and Triage.
- `src/meridian/graph/` — the Sentinel → Triage graph and the SQLite checkpointer wiring.
- `src/meridian/prompts/triage.v1.md` — the versioned triage prompt.
- `src/meridian/run.py` — `python -m meridian.run`. `--fake-llm` is the default. `--live` calls Claude.
- `src/meridian/injector/` — one injector per fault type, and the `python -m meridian.injector` CLI.
- `src/meridian/config.py` — settings loaded from the environment and `.env`. Missing keys do not crash import. `Settings.require_anthropic()`, `Settings.require_xai()`, and `Settings.require_slack()` raise when a caller actually needs those credentials. `--live` asks only for Anthropic.
- `schemas/` — JSON Schema generated from the contracts.
- `docs/adr/` — ADR-001 through ADR-013.
- `.github/workflows/ci.yml` — the checks below.

Generated Parquet under `data/` is gitignored.

## Fault catalog

| Fault | How it is injected | Check that catches it |
| --- | --- | --- |
| `SCHEMA_DRIFT` | Intensity below 0.1 retypes the first key column, below 0.5 renames it, and 0.5 and above drops it. | Column presence and dtype (`column_in_dataframe`, `column_in_schema`, `wrong_dtype`). |
| `NULL_EXPLOSION` | Nulls that fraction of a non-nullable, non-key column. | `not_nullable`. |
| `UNIT_MISMATCH` | Multiplies that fraction of rows by 3.6 on m/s columns, or by 1000 on kilometres. | Range checks, and `speed_matches_distance` on trips. |
| `DUPLICATE_KEYS` | Appends copies of that fraction of rows, repeating the primary key. | `multiple_fields_uniqueness`. |
| `LATE_PARTITION` | Leaves the frame unchanged and moves arrival past the 2 hour SLA. The offset grows with intensity and is always late. | `partition_arrival_sla` on `PartitionMetadata`. |
| `DISTRIBUTION_SHIFT` | Power-warps the PSI column inside its generating range (`u ** (1 + 100 * intensity)`). On trips, distance is recomputed so the speed identity still holds. Legal ranges still hold. | PSI against a quantile profile fit on a separate clean partition. Threshold 0.2. |

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest --cov=src/meridian --cov-fail-under=95 -q -m "not live"
uv run python scripts/export_schemas.py && test -z "$(git status --porcelain schemas/)"
uv run python -m meridian.run --dataset vehicle_telemetry --fault SCHEMA_DRIFT --intensity 0.3 --seed 42 --fake-llm
uv run python -m meridian.run --dataset fleet_daily_kpi --fault NULL_EXPLOSION --intensity 0.0 --seed 7 --fake-llm
```

Copy `.env.example` to `.env` only when a later phase needs credentials. Phase 0 tests do not call external services.

## License

Apache-2.0. Copyright 2026 MERIDIAN contributors.
