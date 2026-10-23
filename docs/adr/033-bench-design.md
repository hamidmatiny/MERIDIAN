# ADR-033: MERIDIAN-Bench splits, the frozen test set, and the false-fix rate

## Context

The scorer modules existed without a harness, so there was no reproducible number for
detection, root cause, the fix, or the critic. The number that matters most is how often
a wrong fix still passes verification. ADR-031 is the wall watchdog and ADR-032 is the
recorded timeline. This is the next free number. The requested "ADR-031" for the bench
is this document.

## Decision

`bench/splits.yaml` is committed and frozen.

- dev: 6 faults × vehicle_telemetry × intensity 0.1 × seed 101 (6 runs).
- test: 6 faults × 3 datasets × intensities 0.1 and 0.5 × seeds 201 and 202 (72 runs).
- hard: clean noisy partitions × 3 datasets × seeds 301–310 (30 runs).

Dev and test seeds are disjoint, and the loader refuses an overlap. The test split
refuses to run without `--final`. A final run writes `LOCK.json` with the commit, prompt
versions, and models. A second final for the same prompt versions requires `--force` and
is flagged in the report.

Each run is the full graph with no Slack. The approval gate records `BENCH_NO_HUMAN` and
does not apply. Results land in `bench/results/<bench_id>/<run_key>.json`. `--resume`
skips completed keys. The default is one run at a time (`MERIDIAN_BENCH_PARALLEL=1`).
Ollama serves one request at a time unless `OLLAMA_NUM_PARALLEL` is at least that
parallelism; the two resident models already fill the memory cap, so the default stays 1.

The headline safety metric is the false-fix rate: verified and not
`repaired_matches_clean`, over verified. Reports state n/N. Rates are not an accuracy
when no ground truth exists (triage severity).

fix.v1 is the committed prompt. Later iterations may change only the fix prompt, the tool
guidance, and the structured feedback, and only from dev traces. The test split is run
once after the version is chosen. The iteration log below is filled from dev results
before that test run.

## Iteration log

Dev only. The test split had not been run when this was written.

A first dev pass was discarded. The capability probe built one abort gate and one
deadline and every run reused them, so the last two runs aborted in about 9 seconds.
That pass is not the baseline. Its detection was 6/6, causal accuracy 4/4,
kind_appropriate 2/3, verified 2/6, false-fix 0/2, escalation 1/6. The harness now
gives each run a new gate, clock, and deadline. The numbers below are the rerun.

| iteration | kind_appropriate | verified | false-fix | escalation | fix structured |
| --- | --- | --- | --- | --- | --- |
| fix.v1 | 3/6 | 4/6 | 0/4 | 2/6 | 7/17 |
| fix.v2 | 6/6 | 3/6 | 0/3 | 1/6 | 11/11 |

v1 traces: two cases escalated after submit_patch failed to validate, and one case
chose RERUN_UPSTREAM when the hypothesis was not about a late arrival. v2 tells the
model to end on submit_patch, to call sample_failures before a DATA_REPAIR, to reserve
RERUN_UPSTREAM for a missed deadline, and it feeds the previous kind back as labeled
lines. No ground-truth fault name was added.

v2 is the version taken to the test split. False-fix stayed 0. Kind and structured
submission both improved. Verified fell by one case. A further prompt edit would be a
guess: the three unverified v2 cases did submit a patch. That is one iteration after
the baseline, inside the cap of three.

## Consequences

A test number cannot be refreshed without `--force`, and the report says so. Dev work
cannot silently reuse a test seed. A bench run does not post to Slack and does not write
an applied remediation.

## Alternatives

Score only the verifier. Rejected: a verified patch can still disagree with the clean
frame, and that disagreement is the safety metric. Tune the prompt on the test split.
Rejected: the test split is one shot.
