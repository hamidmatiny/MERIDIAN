# MERIDIAN-Bench test

- commit: `c56af95d45cedb380a803d820a5d2dd4058fc422`
- models: `{'triage': 'qwen3.5:27b-q4_K_M', 'root_cause': 'qwen3.5:27b-q4_K_M', 'fix': 'qwen3.5:27b-q4_K_M', 'critic': 'gemma4:12b'}`
- prompt versions: `{'triage': 'v1', 'root_cause': 'v1', 'fix': 'v2', 'fix_default': 'v1', 'critic': 'v1'}`
- ollama: `ollama version is 0.40.6`
- seeds and split: `[201, 202]` / test
- forced second final: False

## Headline

| metric | value |
| --- | --- |
| detection recall | 72/72 |
| hard-mode FP rate | 0/0 |
| raw causal accuracy | 71/72 |
| guarded causal accuracy | 71/72 |
| kind_appropriate | 68/72 |
| verified | 27/72 |
| false-fix rate | 0/27 |
| escalation rate | 17/72 |
| critic confusion matrix | approved 33/16; rejected 1/9 |
| machine time p50 | 288.367448 |
| machine time p95 | 498.95234 |
| USD per run | 0.0 |

## Dev and test

| metric | dev-fix-v1 | dev-fix-v2 | test current |
| --- | --- | --- | --- |
| detection_recall | 6/6 | 6/6 | 72/72 |
| raw_causal_correct | 6/6 | 6/6 | 71/72 |
| causal_event_correct | 6/6 | 6/6 | 71/72 |
| kind_appropriate | 3/6 | 6/6 | 68/72 |
| verified | 4/6 | 3/6 | 27/72 |
| false_fix | 0/4 | 0/3 | 0/27 |
| escalation | 2/6 | 1/6 | 17/72 |
| machine_time_p50 | 241.808319 | 301.392669 | 288.367448 |
| machine_time_p95 | 317.57882 | 412.32216 | 498.95234 |
| usd_per_run | 0.0 | 0.0 | 0.0 |

## Detection

- recall: 72/72
- hard-mode false-positive runs: 0/0
- hard-mode incident count: 0

### Recall by fault

- DISTRIBUTION_SHIFT: 12/12
- DUPLICATE_KEYS: 12/12
- LATE_PARTITION: 12/12
- NULL_EXPLOSION: 12/12
- SCHEMA_DRIFT: 12/12
- UNIT_MISMATCH: 12/12

### Recall by dataset

- fleet_daily_kpi: 24/24
- trip_summary: 24/24
- vehicle_telemetry: 24/24

### Precision by fault

- DISTRIBUTION_SHIFT: 12/12
- DUPLICATE_KEYS: 12/12
- LATE_PARTITION: 12/12
- NULL_EXPLOSION: 12/12
- SCHEMA_DRIFT: 12/12
- UNIT_MISMATCH: 12/12

## Triage

No ground-truth severity exists. This is not an accuracy.
- floor-raise: 0/72
- severity distribution: `{'SEV2': 35, 'SEV3': 25, 'SEV1': 4, 'SEV4': 8}`

## Root cause

- raw_causal_correct: 71/72
- causal_event_correct: 71/72
- guardrail false-negative: 0/72
- mean columns_jaccard: 0.9305555555555556
- mean grounding violations: 0.0
- evidence candidates by kind and dataset: `{'1': 48, '2': 24}`
- runs with exactly one candidate: 48/72
- causal accuracy reflects an easy evidence world

## Fix

- kind_appropriate: 68/72
- verified: 27/72
- escalation: 17/72
- repaired_matches_clean among verified: 27/27
- FALSE-FIX RATE: 0/27
- match not re-scored: 0

## Outcomes

- VERIFIED_AWAITING_APPROVAL: 27
- AGENT_ESCALATED: 17
- CRITIC_LIMIT: 2
- FIX_ATTEMPT_LIMIT: 26
- WALL_BUDGET: 0
- NO_PATCH: 0
- OTHER: 0
- sum: 72

## Infra

- runs with infra faults: 0/72
- of those, outcomes differ: 0/0
- outcomes unmeasured: 0
- none

## Critic

Good means `repaired_matches_clean` from score_fix.

| | good | bad |
| --- | --- | --- |
| approved | 33 | 16 |
| rejected | 1 | 9 |

- degraded: 0/59
- corrective: 6/59
- verdicts with no re-scored match: 0
- bad-patch catch rate: 9/25
- approval precision: 33/49
- bad approved patches blocked by the verifier: 16

## Ops

- machine_time_s p50: 288.367448
- machine_time_s p95: 498.95234
- machine_time_s n: 72
- tokens per run: 55024.11111111111
- USD per run: 0.0
- timeouts: 0
- WALL_BUDGET aborts: 0/72
- per-node latency ms: `{'critic': {'p50': 4285.0, 'p95': 11395.0}, 'fix': {'p50': 9354.0, 'p95': 61631.0}, 'root_cause': {'p50': 9847.0, 'p95': 23727.0}, 'triage': {'p50': 23043.0, 'p95': 34411.0}}`
- structured_success_rate: `{'critic': {'n': 95, 'N': 101, 'text': '95/101'}, 'fix': {'n': 147, 'N': 149, 'text': '147/149'}, 'root_cause': {'n': 72, 'N': 72, 'text': '72/72'}, 'triage': {'n': 72, 'N': 72, 'text': '72/72'}}`

