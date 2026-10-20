# ADR-009: InjectedFault as benchmark ground truth

- Status: Accepted

## Context

Phase 6 will score detections. That score is meaningless if the label is reconstructed after the fact from whatever the contract happened to flag. The injection has to record what was done, before evaluation runs.

## Decision

Each injector is a pure function, `inject_<fault>(df, rng, intensity)`, and returns the corrupted frame plus an `InjectedFault`. The label is a frozen contract: `id`, `fault_type`, `dataset`, `columns_affected`, `rows_affected`, `intensity`, and `seed`. The id is a UUID5 of those inputs, so the same call rebuilds the same label. The CLI writes the frame as Parquet and the label as `injected_fault.json`.

`LATE_PARTITION` does not change rows. The frame is copied unchanged and the label says zero rows were affected. The late arrival is `freshness.late_arrival(partition_date, intensity)`, which is always past the SLA. The other faults edit the frame: retype, rename, or drop a key column; null a non-key column; scale m/s by 3.6 or km by 1000; append duplicate primary keys; or power-warp the PSI column inside its generating range. On `trip_summary` the injector also rewrites `distance_km` from the warped speed so the unit identity stays true and the only failing check is PSI.

## Consequences

- A detection is correct when `Incident.fault_type` equals `InjectedFault.fault_type`. The matrix test is that comparison.
- The label does not include a wall-clock time, so two runs with the same seed match.
- Intensity means a different thing per fault (fraction of rows, or which schema edit, or how hard the warp is, or how far past the SLA). The catalog of those meanings lives in the README. The label stores the number either way.
- Phase 6 can read `injected_fault.json` without importing the injector.

## Alternatives considered

- Treating the pandera error text as the label. Rejected because that makes the detector and the answer the same object, so the benchmark cannot see a miss.
- A free-form JSON blob. Rejected. `InjectedFault` is a contract, exported with the other schemas, and rejected when a field is the wrong type.
- Hiding the seed and rebuilding labels by hashing the Parquet bytes. Rejected because late partitions do not change the Parquet file, and two different faults could theoretically collide.
