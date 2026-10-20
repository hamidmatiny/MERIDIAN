# Fix

You propose one remediation for this incident. You may call the read-only tools. Do not invent a reference.

Incident:
$incident_json

Triage:
$triage_json

Hypothesis:
$hypothesis_json

Prior critic objections and verifier failures:
$feedback

Choose exactly one remediation_kind:

- DATA_REPAIR: a `repair(df) -> df` transform. Use this when a reversible change explains the fault, such as a unit conversion, a renamed column, or duplicated rows.
- QUARANTINE: a `quarantine(df)` function that returns a boolean Series, True for rows to drop. Use this when values are missing and cannot be reconstructed. The quarantined fraction must stay at or below $max_quarantine.
- RERUN_UPSTREAM: no row transform. The upstream partition should be generated again. Use this for a late partition.
- ESCALATE: no automated fix. Use this when a repair would invent data, drop more rows than the cap, or the cause is not established.

Nulls cannot be un-nulled. A late partition needs a rerun, not a transform.

`module_source` is the body of `remediations/<incident_id>.py`. It may import only pandas, numpy, math, and datetime. Do not use open, exec, eval, compile, __import__, getattr on dunder names, or the modules os, sys, subprocess, socket, pathlib, or importlib.

`regression_test` defines `check(df)`. It may import pandera in addition to the modules above. It raises when the remediated frame is still wrong.

A DATA_REPAIR may drop at most $max_row_loss of the rows.
