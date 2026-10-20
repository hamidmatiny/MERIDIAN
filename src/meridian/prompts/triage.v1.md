# Triage v1

You assign a severity to one data-quality incident.

The deterministic floor is $floor. You may choose a more severe level (SEV1 is the most severe, then SEV2, SEV3, SEV4). A less severe choice is raised to the floor by the system.

Incident-Id: $incident_id
Downstream: $downstream
Incident-Json:
$incident_json

Return the incident id, a severity, a short rationale, the downstream consumers you believe are affected, and a confidence between 0 and 1.
