"""Write JSON Schema documents for every public contract."""

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from meridian.contracts import (
    ApprovalDecision,
    ApprovalOutcome,
    Evidence,
    FaultType,
    Hypothesis,
    Incident,
    MeridianState,
    Patch,
    Postmortem,
    Severity,
    TimelineEvent,
    Verdict,
    VerificationResult,
)

SchemaType = type[BaseModel] | type[Enum]

# Filename stem, then the type whose schema is written to schemas/<stem>.json.
CONTRACT_SCHEMAS: tuple[tuple[str, SchemaType], ...] = (
    ("severity", Severity),
    ("fault_type", FaultType),
    ("incident", Incident),
    ("evidence", Evidence),
    ("hypothesis", Hypothesis),
    ("patch", Patch),
    ("verdict", Verdict),
    ("verification_result", VerificationResult),
    ("approval_outcome", ApprovalOutcome),
    ("approval_decision", ApprovalDecision),
    ("timeline_event", TimelineEvent),
    ("postmortem", Postmortem),
    ("meridian_state", MeridianState),
)


def schema_document(source: SchemaType) -> str:
    """Return canonical JSON Schema text for one contract type."""
    document: dict[str, Any] = TypeAdapter(source).json_schema()
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def export_schemas(directory: Path) -> tuple[Path, ...]:
    """Write one JSON file per contract and delete schema files that are no longer produced."""
    directory.mkdir(parents=True, exist_ok=True)
    expected: set[str] = set()
    written: list[Path] = []
    for stem, source in CONTRACT_SCHEMAS:
        path = directory / f"{stem}.json"
        path.write_text(schema_document(source), encoding="utf-8")
        written.append(path)
        expected.add(path.name)
    for existing in sorted(directory.glob("*.json")):
        if existing.name not in expected:
            existing.unlink()
    return tuple(written)
