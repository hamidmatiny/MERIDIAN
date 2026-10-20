# ADR-007: Synthetic seeded pipeline instead of external data

- Status: Accepted

## Context

MERIDIAN-Bench has to replay the same incidents on every machine. A dump of real fleet telemetry cannot be checked in, and it would not come with a label for which rows were broken. Phase 1 needs a ground truth that is generated, not discovered.

## Decision

The sample pipeline is three synthetic datasets — `vehicle_telemetry`, `trip_summary`, and `fleet_daily_kpi` — drawn from `numpy.random.Generator` with an explicit seed. `generate_dataset(name, seed, rows, partition_date)` returns that many rows of the named grain. Default size is 10,000. Trip rows satisfy the same speed/distance/duration identity an aggregation would, and the KPI rows stay inside their contract, but each call builds its own frame so the row count is exact.

Lineage is declared in `src/meridian/pipeline/lineage.yaml`, not inferred from a query engine. `downstream_of` walks those edges. Parquet is written with fixed settings and is not committed. `data/` and `data2/` are gitignored. A later phase can point the same contracts at an external Parquet path without changing the incident shape.

## Consequences

- The same seed rebuilds the same frame and, after injection, the same Parquet bytes on a given library version.
- The generators are uniform on published intervals, so the PSI reference is the quantile grid of those intervals rather than a snapshot of one seed.
- The datasets are not a literal group-by of one another. A benchmark that needs that join can add it later; the contracts and the lineage edges do not depend on it.
- Clean-data tests are the false-positive gate. If a generator drifts outside a contract, those tests fail.

## Alternatives considered

- Checked-in Parquet extracts. Rejected because they are bulky, hard to relabel, and not reproducible from a seed.
- Downloading a public telemetry set at test time. Rejected because the suite would depend on the network and the file could change.
- Generating only the raw table and deriving the other two with a group-by inside `generate_dataset`. Rejected for Phase 1 because the caller asks for an exact row count at each grain, and a group-by would make the PSI reference depend on that aggregation. The physical check on `trip_summary` still enforces the identity a derivation would have produced.
