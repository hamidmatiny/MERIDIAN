# MERIDIAN-Bench test

- commit: `637750d16f666ba371310e80204ba2c241836900`
- models: `{'critic': 'gemma4:12b', 'fix': 'qwen3.5:27b-q4_K_M', 'root_cause': 'qwen3.5:27b-q4_K_M', 'triage': 'qwen3.5:27b-q4_K_M'}`
- prompt versions: `{'critic': 'v1', 'fix': 'v2', 'fix_default': 'v1', 'root_cause': 'v1', 'triage': 'v1'}`
- ollama: `ollama version is 0.40.6`
- seeds and split: `[201, 202]` / test
- forced second final: False
- every score was computed by scorer version `52154d863706cae02ee103a6b54081314321bf852a95b2d3a3eeb8f15a718c35` of src/meridian/bench/score_fix.py (parent commit `7edbd177c37080edc435cc43df11831aa22b8a20`)
- this test run started from a dirty tree. LOCK.json was not rewritten.

## Headline

| metric | value |
| --- | --- |
| detection recall | 72/72 |
| hard-mode FP rate | 0/0 |
| raw causal accuracy | 72/72 |
| guarded causal accuracy | 72/72 |
| kind_appropriate | 67/72 |
| verified | 27/72 |
| false-fix rate | 0/27 |
| escalation rate | 15/72 |
| critic confusion matrix | approved 38/14; rejected 1/6 |
| machine time p50 | 261.621484 |
| machine time p95 | 665.596556 |
| USD per run | 0.0 |

## Dev and test

| metric | dev-fix-v1 | dev-fix-v2 | test current |
| --- | --- | --- | --- |
| detection_recall | 6/6 | 6/6 | 72/72 |
| raw_causal_correct | 6/6 | 6/6 | 72/72 |
| causal_event_correct | 6/6 | 6/6 | 72/72 |
| kind_appropriate | 3/6 | 6/6 | 67/72 |
| verified | 4/6 | 3/6 | 27/72 |
| false_fix | 0/4 | 0/3 | 0/27 |
| escalation | 2/6 | 1/6 | 15/72 |
| machine_time_p50 | 241.808319 | 301.392669 | 261.621484 |
| machine_time_p95 | 317.57882 | 412.32216 | 665.596556 |
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
- severity distribution: `{'SEV3': 23, 'SEV4': 8, 'SEV2': 37, 'SEV1': 4}`

## Root cause

- raw_causal_correct: 72/72
- causal_event_correct: 72/72
- guardrail false-negative: 0/72
- mean columns_jaccard: 0.9444444444444444
- mean grounding violations: 0.0

## Fix

- kind_appropriate: 67/72
- verified: 27/72
- escalation: 15/72
- repaired_matches_clean among verified: 27/27
- FALSE-FIX RATE: 0/27
- match not re-scored: 0

## Infra

- runs with infra faults: 4/72
- of those, outcomes differ: 0/4
- outcomes unmeasured: 4
- `trip_summary__DUPLICATE_KEYS__i0.1__s202`: attempt_1_ValueError; outcome_differs=None
- `trip_summary__DUPLICATE_KEYS__i0.5__s202`: attempt_1_ValueError; outcome_differs=None
- `trip_summary__SCHEMA_DRIFT__i0.1__s201`: mcp_timeout, broken_pipe; outcome_differs=None
- `vehicle_telemetry__DUPLICATE_KEYS__i0.1__s201`: attempt_1_ValueError; outcome_differs=None

## Critic

Good means `repaired_matches_clean` from score_fix.

| | good | bad |
| --- | --- | --- |
| approved | 38 | 14 |
| rejected | 1 | 6 |

- degraded: 0/59
- corrective: 3/59
- verdicts with no re-scored match: 0

## Ops

- machine_time_s p50: 261.621484
- machine_time_s p95: 665.596556
- machine_time_s n: 72
- tokens per run: 55694.73611111111
- USD per run: 0.0
- timeouts: 0
- WALL_BUDGET aborts: 2/72
- per-node latency ms: `{'critic': {'p50': 4996.0, 'p95': 10380.0}, 'fix': {'p50': 7635.0, 'p95': 52148.0}, 'root_cause': {'p50': 9199.0, 'p95': 37003.0}, 'triage': {'p50': 22656.0, 'p95': 38217.0}}`
- structured_success_rate: `{'critic': {'n': 98, 'N': 102, 'text': '98/102'}, 'fix': {'n': 141, 'N': 143, 'text': '141/143'}, 'root_cause': {'n': 72, 'N': 72, 'text': '72/72'}, 'triage': {'n': 72, 'N': 72, 'text': '72/72'}}`

