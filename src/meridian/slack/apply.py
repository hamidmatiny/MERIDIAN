"""Write an approved remediation to disk. Nothing here pushes git or edits data."""

import json
from pathlib import Path

from meridian.agents.patch_text import source_from_diff
from meridian.contracts.incident_case import IncidentCase


def apply_patch(case: IncidentCase, root: Path) -> Path:
    """Write the module, the regression test, and the approval manifest."""
    if case.patch is None or case.approval is None or case.verification is None:
        raise ValueError("apply requires a patch, a decision, and a verification")
    folder = root / str(case.incident.id)
    folder.mkdir(parents=True, exist_ok=True)
    source = source_from_diff(case.patch.diff)
    if source and not source.endswith("\n"):
        source = f"{source}\n"
    test = case.patch.regression_test
    if test and not test.endswith("\n"):
        test = f"{test}\n"
    (folder / "remediation.py").write_text(source, encoding="utf-8")
    (folder / "test_regression.py").write_text(test, encoding="utf-8")
    manifest = {
        "approver": case.approval.approver,
        "decided_at": case.approval.decided_at.isoformat(),
        "patch_id": str(case.patch.id),
        "verifier_checks": [item.model_dump(mode="json") for item in case.verification.checks],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return folder
