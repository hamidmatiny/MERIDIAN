"""JSON round trips for every contract model."""

from datetime import UTC, date, datetime
from uuid import UUID

from pydantic import BaseModel
from tests.contracts.builders import (
    make_approval,
    make_evidence,
    make_hypothesis,
    make_incident,
    make_patch,
    make_postmortem,
    make_verdict,
    make_verification,
)

from meridian.contracts import (
    ColumnProfile,
    InjectedFault,
    MeridianState,
    PartitionMetadata,
    ReferenceProfile,
)
from meridian.contracts.fault_type import FaultType
from meridian.schema_export import CONTRACT_SCHEMAS


def _round_trip(model: BaseModel) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


def test_each_model_round_trips_through_json() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id, confidence=1)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    models: list[BaseModel] = [
        incident,
        make_evidence(),
        hypothesis,
        patch,
        make_verdict(patch_id=patch.id, round_number=2),
        make_verification(patch_id=patch.id, passed=False, failures_remaining=3),
        make_approval(patch_id=patch.id),
        make_postmortem(incident_id=incident.id),
        MeridianState(
            incident=incident,
            hypothesis=hypothesis,
            patch=patch,
            fix_attempts=1,
            critic_rounds=2,
        ),
        MeridianState(),
        InjectedFault(
            id=UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a11"),
            fault_type=FaultType.NULL_EXPLOSION,
            dataset="vehicle_telemetry",
            columns_affected=["battery_pct"],
            rows_affected=10,
            intensity=0.1,
            seed=42,
        ),
        PartitionMetadata(
            dataset="vehicle_telemetry",
            partition_date=date(2026, 10, 19),
            arrived_at=datetime(2026, 10, 20, 0, 30, tzinfo=UTC),
        ),
        ColumnProfile(
            column="speed_mps",
            bin_edges=[0.0, 1.0],
            expected_proportions=[1.0],
        ),
        ReferenceProfile(
            dataset="vehicle_telemetry",
            psi_threshold=0.2,
            columns=[
                ColumnProfile(
                    column="speed_mps",
                    bin_edges=[0.0, 1.0],
                    expected_proportions=[1.0],
                )
            ],
        ),
    ]
    for model in models:
        _round_trip(model)


def test_round_trip_covers_every_contract_model() -> None:
    incident = make_incident()
    hypothesis = make_hypothesis(incident_id=incident.id)
    patch = make_patch(incident_id=incident.id, hypothesis_id=hypothesis.id)
    instances = {
        "Incident": incident,
        "Evidence": make_evidence(),
        "Hypothesis": hypothesis,
        "Patch": patch,
        "Verdict": make_verdict(patch_id=patch.id),
        "VerificationResult": make_verification(patch_id=patch.id),
        "ApprovalDecision": make_approval(patch_id=patch.id),
        "TimelineEvent": make_postmortem(incident_id=incident.id).timeline[0],
        "Postmortem": make_postmortem(incident_id=incident.id),
        "MeridianState": MeridianState(incident=incident, critic_rounds=1),
        "InjectedFault": InjectedFault(
            id=UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a11"),
            fault_type=FaultType.LATE_PARTITION,
            dataset="trip_summary",
            columns_affected=[],
            rows_affected=0,
            intensity=0.2,
            seed=1,
        ),
        "PartitionMetadata": PartitionMetadata(
            dataset="trip_summary",
            partition_date=date(2026, 10, 19),
            arrived_at=datetime(2026, 10, 20, 0, 30, tzinfo=UTC),
        ),
        "ColumnProfile": ColumnProfile(
            column="speed_mps",
            bin_edges=[20.0, 40.0],
            expected_proportions=[1.0],
        ),
        "ReferenceProfile": ReferenceProfile(
            dataset="vehicle_telemetry",
            psi_threshold=0.2,
            columns=[
                ColumnProfile(
                    column="speed_mps",
                    bin_edges=[20.0, 30.0, 40.0],
                    expected_proportions=[0.5, 0.5],
                )
            ],
        ),
    }
    model_names = {
        source.__name__
        for _stem, source in CONTRACT_SCHEMAS
        if isinstance(source, type) and issubclass(source, BaseModel)
    }
    assert model_names == set(instances)
    for model in instances.values():
        _round_trip(model)
