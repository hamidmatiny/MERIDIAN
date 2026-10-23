# MERIDIAN-Bench dev

- commit: `637750d16f666ba371310e80204ba2c241836900`
- models: `{'critic': 'gemma4:12b', 'fix': 'qwen3.5:27b-q4_K_M', 'root_cause': 'qwen3.5:27b-q4_K_M', 'triage': 'qwen3.5:27b-q4_K_M'}`
- prompt versions: `{'critic': 'v1', 'fix': 'v2', 'fix_default': 'v1', 'root_cause': 'v1', 'triage': 'v1'}`
- ollama: `ollama version is 0.40.6`
- seeds and split: `[101]` / dev
- forced second final: False
- every score was computed by scorer version `52154d863706cae02ee103a6b54081314321bf852a95b2d3a3eeb8f15a718c35` of src/meridian/bench/score_fix.py (parent commit `7edbd177c37080edc435cc43df11831aa22b8a20`)

## Headline

| metric | value |
| --- | --- |
| detection recall | 6/6 |
| hard-mode FP rate | 0/0 |
| raw causal accuracy | 6/6 |
| guarded causal accuracy | 6/6 |
| kind_appropriate | 6/6 |
| verified | 3/6 |
| false-fix rate | 0/3 |
| escalation rate | 1/6 |
| critic confusion matrix | approved 3/1; rejected 0/0 |
| machine time p50 | 301.392669 |
| machine time p95 | 412.32216 |
| USD per run | 0.0 |

## Detection

- recall: 6/6
- hard-mode false-positive runs: 0/0
- hard-mode incident count: 0

### Recall by fault

- DISTRIBUTION_SHIFT: 1/1
- DUPLICATE_KEYS: 1/1
- LATE_PARTITION: 1/1
- NULL_EXPLOSION: 1/1
- SCHEMA_DRIFT: 1/1
- UNIT_MISMATCH: 1/1

### Recall by dataset

- vehicle_telemetry: 6/6

### Precision by fault

- DISTRIBUTION_SHIFT: 1/1
- DUPLICATE_KEYS: 1/1
- LATE_PARTITION: 1/1
- NULL_EXPLOSION: 1/1
- SCHEMA_DRIFT: 1/1
- UNIT_MISMATCH: 1/1

## Triage

No ground-truth severity exists. This is not an accuracy.
- floor-raise: 0/6
- severity distribution: `{'SEV2': 5, 'SEV1': 1}`

## Root cause

- raw_causal_correct: 6/6
- causal_event_correct: 6/6
- guardrail false-negative: 0/6
- mean columns_jaccard: 0.9166666666666666
- mean grounding violations: 0.0

## Fix

- kind_appropriate: 6/6
- verified: 3/6
- escalation: 1/6
- repaired_matches_clean among verified: 3/3
- FALSE-FIX RATE: 0/3
- match not re-scored: 0

## Infra

- runs with infra faults: 0/6
- of those, outcomes differ: 0/0
- outcomes unmeasured: 0
- none

## Critic

Good means `repaired_matches_clean` from score_fix.

| | good | bad |
| --- | --- | --- |
| approved | 3 | 1 |
| rejected | 0 | 0 |

- degraded: 0/4
- corrective: 1/4
- verdicts with no re-scored match: 0

## Ops

- machine_time_s p50: 301.392669
- machine_time_s p95: 412.32216
- machine_time_s n: 6
- tokens per run: 47637.0
- USD per run: 0.0
- timeouts: 0
- WALL_BUDGET aborts: 0/6
- per-node latency ms: `{'critic': {'p50': 3495.0, 'p95': 10041.0}, 'fix': {'p50': 11265.0, 'p95': 86718.0}, 'root_cause': {'p50': 10262.0, 'p95': 27987.0}, 'triage': {'p50': 36038.0, 'p95': 38441.0}}`
- structured_success_rate: `{'critic': {'n': 5, 'N': 6, 'text': '5/6'}, 'fix': {'n': 11, 'N': 11, 'text': '11/11'}, 'root_cause': {'n': 6, 'N': 6, 'text': '6/6'}, 'triage': {'n': 6, 'N': 6, 'text': '6/6'}}`

