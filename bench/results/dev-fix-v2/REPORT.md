# MERIDIAN-Bench dev

- commit: `637750d16f666ba371310e80204ba2c241836900`
- models: `{'triage': 'qwen3.5:27b-q4_K_M', 'root_cause': 'qwen3.5:27b-q4_K_M', 'fix': 'qwen3.5:27b-q4_K_M', 'critic': 'gemma4:12b'}`
- prompt versions: `{'triage': 'v1', 'root_cause': 'v1', 'fix': 'v2', 'fix_default': 'v1', 'critic': 'v1'}`
- ollama: `ollama version is 0.40.6`
- seeds and split: `[101]` / dev
- forced second final: False

## Headline

| metric | value |
| --- | --- |
| detection recall | 6/6 |
| hard-mode FP rate | 0/0 |
| causal accuracy | 6/6 |
| kind_appropriate | 6/6 |
| verified | 3/6 |
| false-fix rate | 0/3 |
| escalation rate | 1/6 |
| machine time p50 | 301.392669 |
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

## Critic

Good means `repaired_matches_clean` from score_fix.

| | good | bad |
| --- | --- | --- |
| approved | 3 | 1 |
| rejected | 0 | 0 |

- degraded: 0/4
- corrective: 1/4

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

