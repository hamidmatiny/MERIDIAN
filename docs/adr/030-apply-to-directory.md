# ADR-030: Apply an approval into a directory, not a merge

- Status: Accepted

## Context

ADR-004 says a patch does not merge without an `ApprovalDecision`. Phase 5 has to put the approved remediation somewhere a person can see, without pushing git and without rewriting the partition. Rejected, timed-out, and acknowledged cases still need a postmortem.

## Decision

`APPROVED` writes `remediations/applied/<incident_id>/`. The directory holds the remediation module, the regression test, and `manifest.json` (approver, decision time, patch id, verifier checks). The source is the patch diff. The partition files are not modified. Git is not pushed.

`REJECTED`, `TIMEOUT`, and `ACKNOWLEDGED` write nothing in that directory.

Every finished case writes `docs/postmortems/<incident_id>.md` from a fixed template and posts that markdown on the approval thread. The template lists the timeline (detected, triaged, hypothesis, each attempt, the decision), the root cause, the fix summary, the resolution time, the token counts, and the USD from the budget ledger. Free-tier calls are $0.00. No model writes the report.

## Consequences

- A person can read the approved files and commit them later. The system does not commit them.
- A rejection leaves no remediation directory. The postmortem still records the comment.
- The postmortem is reproducible from the case. Running it twice overwrites the same file with the same text.
- The applied directory is not a merge, so a later review can still refuse the files.

## Alternatives considered

- Opening a pull request and merging on Approve. Rejected for this phase. The gate's job is the decision and the files, not repository write access.
- Mutating the failing parquet in place. Rejected. The verifier already replayed the patch. The source data stays as the incident found it.
- Asking a model to draft the postmortem. Rejected. The numbers are already on the case, and a template does not drift.
