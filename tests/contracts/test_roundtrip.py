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
    ChangeEvent,
    ColumnComparison,
    ColumnDescription,
    ColumnProfile,
    ContractSummary,
    EventPage,
    EventView,
    FailureSample,
    GroundingViolation,
    IncidentCase,
    InjectedFault,
    LineageResult,
    MeridianState,
    Objection,
    ObjectionKind,
    PartitionDescription,
    PartitionMetadata,
    ReferenceComparison,
    ReferenceProfile,
    ToolCall,
    VerificationCheck,
)
from meridian.contracts.change_event import EventKind
from meridian.contracts.fault_type import FaultType
from meridian.contracts.llm_call import LLMCall
from meridian.contracts.partition_ref import PartitionRef
from meridian.contracts.severity import Severity
from meridian.contracts.triage import TriageAssessment, TriageResult
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
            run_id=UUID("11111111-1111-4111-8111-111111111111"),
            cases={
                str(incident.id): IncidentCase(
                    incident=incident,
                    hypothesis=hypothesis,
                    patch=patch,
                    fix_attempts=1,
                    critic_rounds=2,
                )
            },
        ),
        MeridianState(run_id=UUID("22222222-2222-4222-8222-222222222222")),
        InjectedFault(
            id=UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a11"),
            fault_type=FaultType.NULL_EXPLOSION,
            dataset="vehicle_telemetry",
            columns_affected=["battery_pct"],
            rows_affected=10,
            intensity=0.1,
            seed=42,
            causal_event_id=UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a12"),
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
        PartitionRef(
            dataset="vehicle_telemetry",
            partition_date=date(2026, 10, 19),
            parquet_path="data/vehicle_telemetry.parquet",
            metadata=PartitionMetadata(
                dataset="vehicle_telemetry",
                partition_date=date(2026, 10, 19),
                arrived_at=datetime(2026, 10, 20, 0, 30, tzinfo=UTC),
            ),
            reference_profile_path="data/reference_profile.json",
        ),
        TriageAssessment(
            incident_id=incident.id,
            severity=Severity.SEV4,
            rationale="low",
            affected_consumers=["trip_summary"],
            confidence=0.25,
        ),
        TriageResult(
            incident_id=incident.id,
            severity=Severity.SEV2,
            rationale="raised to the floor",
            affected_consumers=["trip_summary"],
            confidence=0.25,
            llm_severity=Severity.SEV4,
            floor_severity=Severity.SEV2,
            final_severity=Severity.SEV2,
            prompt_version="v1",
        ),
        LLMCall(
            node="triage",
            model="fake-triage",
            prompt_version="v1",
            input_tokens=12,
            output_tokens=8,
            latency_ms=3,
            attempt=1,
            succeeded=True,
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
        "Objection": Objection(kind=ObjectionKind.OTHER, message="revise the patch"),
        "VerificationCheck": VerificationCheck(name="contracts", passed=True),
        "VerificationResult": make_verification(patch_id=patch.id),
        "ApprovalDecision": make_approval(patch_id=patch.id),
        "TimelineEvent": make_postmortem(incident_id=incident.id).timeline[0],
        "Postmortem": make_postmortem(incident_id=incident.id),
        "IncidentCase": IncidentCase(
            incident=incident,
            hypothesis=hypothesis,
            patch=patch,
            verdicts=[make_verdict(patch_id=patch.id)],
            fix_attempts=1,
            critic_rounds=1,
        ),
        "MeridianState": MeridianState(
            run_id=UUID("33333333-3333-4333-8333-333333333333"),
            cases={str(incident.id): IncidentCase(incident=incident, critic_rounds=1)},
        ),
        "InjectedFault": InjectedFault(
            id=UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a11"),
            fault_type=FaultType.LATE_PARTITION,
            dataset="trip_summary",
            columns_affected=[],
            rows_affected=0,
            intensity=0.2,
            seed=1,
            arrival_delay_s=17_280,
            causal_event_id=UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a12"),
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
        "PartitionRef": PartitionRef(
            dataset="fleet_daily_kpi",
            partition_date=date(2026, 10, 19),
            parquet_path="data/fleet.parquet",
            metadata=PartitionMetadata(
                dataset="fleet_daily_kpi",
                partition_date=date(2026, 10, 19),
                arrived_at=datetime(2026, 10, 20, 0, 30, tzinfo=UTC),
            ),
            reference_profile_path="data/profile.json",
        ),
        "TriageAssessment": TriageAssessment(
            incident_id=incident.id,
            severity=Severity.SEV3,
            rationale="watch",
            affected_consumers=[],
            confidence=0.5,
        ),
        "TriageResult": TriageResult(
            incident_id=incident.id,
            severity=Severity.SEV3,
            rationale="watch",
            affected_consumers=[],
            confidence=0.5,
            llm_severity=Severity.SEV3,
            floor_severity=Severity.SEV4,
            final_severity=Severity.SEV3,
            prompt_version="v1",
        ),
        "LLMCall": LLMCall(
            node="triage",
            model="fake-triage",
            prompt_version="v1",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            attempt=1,
            succeeded=False,
        ),
        "ChangeEvent": ChangeEvent(
            id=UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a12"),
            ts=datetime(2026, 10, 19, 15, 0, tzinfo=UTC),
            kind=EventKind.SCHEMA_CHANGE,
            dataset="vehicle_telemetry",
            author="pipeline-bot",
            summary="Causal event: renamed speed_mps",
            diff_excerpt="- speed_mps",
        ),
        "ToolCall": ToolCall(
            node="root_cause",
            tool="get_event",
            args_hash="abc",
            ref_count=1,
            latency_ms=1,
            ok=True,
        ),
        "LineageResult": LineageResult(
            ref="lineage:upstream:trip_summary",
            dataset="trip_summary",
            direction="upstream",
            datasets=["vehicle_telemetry"],
        ),
        "ContractSummary": ContractSummary(
            ref="contract:vehicle_telemetry",
            dataset="vehicle_telemetry",
            contract_name="VehicleTelemetry",
            columns=["vehicle_id"],
            checks=[],
        ),
        "ColumnDescription": ColumnDescription(
            column="speed_mps",
            dtype="float64",
            null_rate=0.0,
            ref="column:/tmp/part.parquet:speed_mps",
        ),
        "PartitionDescription": PartitionDescription(
            ref="partition:/tmp/part.parquet",
            path="/tmp/part.parquet",
            row_count=1,
            columns=[
                ColumnDescription(
                    column="speed_mps",
                    dtype="float64",
                    null_rate=0.0,
                    ref="column:/tmp/part.parquet:speed_mps",
                )
            ],
        ),
        "ColumnComparison": ColumnComparison(
            column="speed_mps",
            psi=0.1,
            null_rate=0.0,
            ref="psi:vehicle_telemetry:speed_mps",
        ),
        "ReferenceComparison": ReferenceComparison(
            ref="psi:vehicle_telemetry",
            dataset="vehicle_telemetry",
            columns=[
                ColumnComparison(
                    column="speed_mps",
                    psi=None,
                    null_rate=0.0,
                    ref="psi:vehicle_telemetry:speed_mps",
                )
            ],
        ),
        "FailureSample": FailureSample(
            ref="sample:11111111-1111-4111-8111-111111111111",
            incident_id=UUID("11111111-1111-4111-8111-111111111111"),
            rows=[{"speed_mps": 1.0}],
        ),
        "EventView": EventView(
            id=UUID("6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a12"),
            ts=datetime(2026, 10, 19, 15, 0, tzinfo=UTC),
            kind=EventKind.CONFIG_CHANGE,
            dataset="vehicle_telemetry",
            author="ada",
            summary="Distractor: routine CONFIG_CHANGE",
            diff_excerpt="unrelated",
            ref="event:6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a12",
        ),
        "GroundingViolation": GroundingViolation(
            field="causal_event_id",
            raw="event:6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a12",
            normalized="event:6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a12",
            allowed_refs_sample=["event:6f9c0b3e-3c1a-4f5e-9a4b-8e2d1c0b7a12"],
        ),
        "EventPage": EventPage(
            ref="events:vehicle_telemetry",
            events=[],
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
