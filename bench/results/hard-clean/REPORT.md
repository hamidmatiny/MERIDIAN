# MERIDIAN-Bench hard

- commit: `7e7f9f5e80eb1a0ebf779b6bc815be7a52b40afa`
- models: `{'triage': 'qwen3.5:27b-q4_K_M', 'root_cause': 'qwen3.5:27b-q4_K_M', 'fix': 'qwen3.5:27b-q4_K_M', 'critic': 'gemma4:12b'}`
- prompt versions: `{'triage': 'v1', 'root_cause': 'v1', 'fix': 'v2', 'fix_default': 'v1', 'critic': 'v1'}`
- ollama: `ollama version is 0.40.6`
- seeds and split: `[301, 302, 303, 304, 305, 306, 307, 308, 309, 310]` / hard
- forced second final: False

## Headline

| metric | value |
| --- | --- |
| detection recall | 0/0 |
| hard-mode FP rate | 0/30 |
| raw causal accuracy | 0/0 |
| guarded causal accuracy | 0/0 |
| kind_appropriate | 0/0 |
| verified | 0/0 |
| false-fix rate | 0/0 |
| escalation rate | 0/0 |
| critic confusion matrix | approved 0/0; rejected 0/0 |
| machine time p50 | None |
| machine time p95 | None |
| USD per run | 0.0 |

## Detection

- recall: 0/0
- hard-mode false-positive runs: 0/30
- hard-mode incident count: 0

### Recall by fault

- none

### Recall by dataset

- none

### Precision by fault

- none

## Triage

No ground-truth severity exists. This is not an accuracy.
- floor-raise: 0/0
- severity distribution: `{}`

## Root cause

- raw_causal_correct: 0/0
- causal_event_correct: 0/0
- guardrail false-negative: 0/0
- mean columns_jaccard: None
- mean grounding violations: None
- evidence candidates by kind and dataset: `{}`
- runs with exactly one candidate: 0/0

## Fix

- kind_appropriate: 0/0
- verified: 0/0
- escalation: 0/0
- repaired_matches_clean among verified: 0/0
- FALSE-FIX RATE: 0/0
- match not re-scored: 0

## Outcomes

- VERIFIED_AWAITING_APPROVAL: 0
- AGENT_ESCALATED: 0
- CRITIC_LIMIT: 0
- FIX_ATTEMPT_LIMIT: 0
- WALL_BUDGET: 0
- NO_PATCH: 0
- OTHER: 30
- sum: 30
- OTHER `vehicle_telemetry__clean__s301`: no incident
- OTHER `vehicle_telemetry__clean__s302`: no incident
- OTHER `vehicle_telemetry__clean__s303`: no incident
- OTHER `vehicle_telemetry__clean__s304`: no incident
- OTHER `vehicle_telemetry__clean__s305`: no incident
- OTHER `vehicle_telemetry__clean__s306`: no incident
- OTHER `vehicle_telemetry__clean__s307`: no incident
- OTHER `vehicle_telemetry__clean__s308`: no incident
- OTHER `vehicle_telemetry__clean__s309`: no incident
- OTHER `vehicle_telemetry__clean__s310`: no incident
- OTHER `trip_summary__clean__s301`: no incident
- OTHER `trip_summary__clean__s302`: no incident
- OTHER `trip_summary__clean__s303`: no incident
- OTHER `trip_summary__clean__s304`: no incident
- OTHER `trip_summary__clean__s305`: no incident
- OTHER `trip_summary__clean__s306`: no incident
- OTHER `trip_summary__clean__s307`: no incident
- OTHER `trip_summary__clean__s308`: no incident
- OTHER `trip_summary__clean__s309`: no incident
- OTHER `trip_summary__clean__s310`: no incident
- OTHER `fleet_daily_kpi__clean__s301`: no incident
- OTHER `fleet_daily_kpi__clean__s302`: no incident
- OTHER `fleet_daily_kpi__clean__s303`: no incident
- OTHER `fleet_daily_kpi__clean__s304`: no incident
- OTHER `fleet_daily_kpi__clean__s305`: no incident
- OTHER `fleet_daily_kpi__clean__s306`: no incident
- OTHER `fleet_daily_kpi__clean__s307`: no incident
- OTHER `fleet_daily_kpi__clean__s308`: no incident
- OTHER `fleet_daily_kpi__clean__s309`: no incident
- OTHER `fleet_daily_kpi__clean__s310`: no incident

## Infra

- runs with infra faults: 0/30
- of those, outcomes differ: 0/0
- outcomes unmeasured: 0
- none

## Critic

Good means `repaired_matches_clean` from score_fix.

| | good | bad |
| --- | --- | --- |
| approved | 0 | 0 |
| rejected | 0 | 0 |

- degraded: 0/0
- corrective: 0/0
- verdicts with no re-scored match: 0
- bad-patch catch rate: 0/0
- approval precision: 0/0
- bad approved patches blocked by the verifier: 0

## Ops

- machine_time_s p50: None
- machine_time_s p95: None
- machine_time_s n: 0
- tokens per run: 0.0
- USD per run: 0.0
- timeouts: 0
- WALL_BUDGET aborts: 0/30
- per-node latency ms: `{}`
- structured_success_rate: `{}`

