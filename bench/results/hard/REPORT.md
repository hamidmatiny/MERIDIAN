# MERIDIAN-Bench hard

- commit: `637750d16f666ba371310e80204ba2c241836900`
- models: `{'triage': 'qwen3.5:27b-q4_K_M', 'root_cause': 'qwen3.5:27b-q4_K_M', 'fix': 'qwen3.5:27b-q4_K_M', 'critic': 'gemma4:12b'}`
- prompt versions: `{'triage': 'v1', 'root_cause': 'v1', 'fix': 'v1', 'fix_default': 'v1', 'critic': 'v1'}`
- ollama: `ollama version is 0.40.6`
- seeds and split: `[301, 302, 303, 304, 305, 306, 307, 308, 309, 310]` / hard
- forced second final: False

## Headline

| metric | value |
| --- | --- |
| detection recall | 0/0 |
| hard-mode FP rate | 0/30 |
| causal accuracy | 0/0 |
| kind_appropriate | 0/0 |
| verified | 0/0 |
| false-fix rate | 0/0 |
| escalation rate | 0/0 |
| machine time p50 | None |
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

## Critic

Good means `repaired_matches_clean` from score_fix.

| | good | bad |
| --- | --- | --- |
| approved | 0 | 0 |
| rejected | 0 | 0 |

- degraded: 0/0
- corrective: 0/0

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

