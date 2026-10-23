# MERIDIAN-Bench dev

- commit: `637750d16f666ba371310e80204ba2c241836900`
- models: `{'triage': 'qwen3.5:27b-q4_K_M', 'root_cause': 'qwen3.5:27b-q4_K_M', 'fix': 'qwen3.5:27b-q4_K_M', 'critic': 'gemma4:12b'}`
- prompt versions: `{'triage': 'v1', 'root_cause': 'v1', 'fix': 'v1', 'fix_default': 'v1', 'critic': 'v1'}`
- ollama: `ollama version is 0.40.6`
- seeds and split: `[101]` / dev
- forced second final: False

## Headline

| metric | value |
| --- | --- |
| detection recall | 6/6 |
| hard-mode FP rate | 0/0 |
| causal accuracy | 6/6 |
| kind_appropriate | 3/6 |
| verified | 4/6 |
| false-fix rate | 0/4 |
| escalation rate | 2/6 |
| machine time p50 | 241.808319 |
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

- kind_appropriate: 3/6
- verified: 4/6
- escalation: 2/6
- repaired_matches_clean among verified: 4/4
- FALSE-FIX RATE: 0/4

## Critic

Good means `repaired_matches_clean` from score_fix.

| | good | bad |
| --- | --- | --- |
| approved | 4 | 0 |
| rejected | 0 | 2 |

- degraded: 0/6
- corrective: 1/6

## Ops

- machine_time_s p50: 241.808319
- machine_time_s p95: 317.57882
- machine_time_s n: 6
- tokens per run: 54395.0
- USD per run: 0.0
- timeouts: 0
- WALL_BUDGET aborts: 0/6
- per-node latency ms: `{'critic': {'p50': 8908.0, 'p95': 11897.0}, 'fix': {'p50': 7876.0, 'p95': 41047.0}, 'root_cause': {'p50': 10431.0, 'p95': 28098.0}, 'triage': {'p50': 35854.0, 'p95': 38116.0}}`
- structured_success_rate: `{'critic': {'n': 8, 'N': 9, 'text': '8/9'}, 'fix': {'n': 7, 'N': 17, 'text': '7/17'}, 'root_cause': {'n': 6, 'N': 6, 'text': '6/6'}, 'triage': {'n': 6, 'N': 6, 'text': '6/6'}}`

