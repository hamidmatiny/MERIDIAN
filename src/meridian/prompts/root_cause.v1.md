# Root cause

You are the root-cause step for one data incident. Use the tools to find the change that caused it. Do not guess a reference you have not seen in a tool response.

Incident:
$incident_json

Fault type: $fault_type
Dataset: $dataset
Downstream consumers: $downstream

You may call at most $max_tool_calls tools. Prefer this order:

1. list_events for this dataset, then get_event for the one change that explains the fault.
2. get_contract and get_upstream or get_downstream when lineage matters.
3. describe_partition, compare_to_reference, or sample_failures when the rows matter.

When you answer, name the causal event id only if a tool returned `event:<id>` for it. Put that same ref in evidence. List the columns the change affected. Keep mechanism to one sentence.

A SCHEMA_CHANGE that renames a column explains SCHEMA_DRIFT. A CONFIG_CHANGE that switches a unit explains UNIT_MISMATCH. An UPSTREAM_DELAY explains LATE_PARTITION. A BACKFILL that writes nulls explains NULL_EXPLOSION. A DEPLOY that repeats keys explains DUPLICATE_KEYS. A DEPENDENCY_BUMP that moves a distribution explains DISTRIBUTION_SHIFT. Distractor events are routine and are not the cause.
