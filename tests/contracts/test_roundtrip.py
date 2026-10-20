"""JSON round trips for every contract model."""

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

from meridian.contracts import MeridianState
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
    }
    model_names = {
        source.__name__
        for _stem, source in CONTRACT_SCHEMAS
        if isinstance(source, type) and issubclass(source, BaseModel)
    }
    assert model_names == set(instances)
    for model in instances.values():
        _round_trip(model)
