# Fix

You propose one remediation for this incident. You may call the read-only tools. Do not invent a reference.

Incident:
$incident_json

Triage:
$triage_json

Hypothesis:
$hypothesis_json

Prior attempt, one field per line:
$feedback

Choose exactly one remediation_kind:

- DATA_REPAIR: a `repair(df) -> df` transform. Use this when a reversible change explains the fault, such as a unit conversion, a renamed column, or duplicated rows. Call sample_failures before submit_patch. A DATA_REPAIR submitted without that call is refused.
- QUARANTINE: a `quarantine(df)` function that returns a boolean Series, True for rows to drop. Use this when values are missing and cannot be reconstructed. The quarantined fraction must stay at or below $max_quarantine.
- RERUN_UPSTREAM: no row transform. Use this only when the hypothesis says the partition arrived after its deadline. Do not use it when values shifted, units changed, or keys were repeated.
- ESCALATE: no automated fix. Use this when a repair would invent data, drop more rows than the cap, or the cause is not established. Do not escalate because a tool call was hard to format. A missing submit_patch is recorded as ESCALATE.

Nulls cannot be un-nulled. A late partition needs a rerun, not a transform.

Your last action must be the submit_patch tool. A prose reply is not a remediation. submit_patch takes remediation_kind, module_source, and regression_test, all three, every time.

`module_source` is the body of `remediations/<incident_id>.py`. It may import only pandas, numpy, math, and datetime. Do not use open, exec, eval, compile, __import__, getattr on dunder names, or the modules os, sys, subprocess, socket, pathlib, or importlib.

`regression_test` defines `check(df)`. It may import pandera in addition to the modules above. It raises when the remediated frame is still wrong.

A DATA_REPAIR may drop at most $max_row_loss of the rows.

If the feedback names a previous_kind and an objection, change the kind or the function. Do not resubmit the same body.
