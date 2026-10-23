# MERIDIAN-Bench hard

- commit: `637750d16f666ba371310e80204ba2c241836900`
- models: `{'critic': 'gemma4:12b', 'fix': 'qwen3.5:27b-q4_K_M', 'root_cause': 'qwen3.5:27b-q4_K_M', 'triage': 'qwen3.5:27b-q4_K_M'}`
- prompt versions: `{'critic': 'v1', 'fix': 'v1', 'fix_default': 'v1', 'root_cause': 'v1', 'triage': 'v1'}`
- ollama: `ollama version is 0.40.6`
- seeds and split: `[301, 302, 303, 304, 305, 306, 307, 308, 309, 310]` / hard
- forced second final: False
- every score was computed by scorer version `52154d863706cae02ee103a6b54081314321bf852a95b2d3a3eeb8f15a718c35` of src/meridian/bench/score_fix.py (parent commit `7edbd177c37080edc435cc43df11831aa22b8a20`)

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

## Fix

- kind_appropriate: 0/0
- verified: 0/0
- escalation: 0/0
- repaired_matches_clean among verified: 0/0
- FALSE-FIX RATE: 0/0
- match not re-scored: 0

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

