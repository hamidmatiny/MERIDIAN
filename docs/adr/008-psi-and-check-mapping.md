# ADR-008: PSI for distribution shift, and an explicit check-to-FaultType map

- Status: Accepted

## Context

A shifted numeric column can still sit inside its legal range, so `speed_mps` between 0 and 60 does not tell Sentinel that the shape changed. The other five fault types already have a direct pandera or metadata check. Something has to turn each of those failures into exactly one `FaultType`, including the shift.

## Decision

Distribution shift is the Population Stability Index against a quantile profile fit from a held-out clean partition. `fit_reference_profile` reads numeric columns on that frame and sets bin edges to its quantiles. It does not read the generator's bounds. The profile used to score a partition is fit on a different seed. The default threshold stays 0.2 and is configurable on the `ReferenceProfile`. Values outside the fitted edges are ignored by the histogram, so a unit conversion that leaves the reference range is not also scored as a shift. `quality/` does not import the generator.

Freshness is separate from pandera. `PartitionMetadata.arrived_at` is late when it is after partition close plus the SLA. The default SLA is 2 hours.

`src/meridian/quality/mapping.py` is the only map from a check name or a pandera reason code to a `FaultType`. Lookup order is fixed: a `check_error` is schema drift, then the check-name table, then the reason-code table. An unknown pair raises. There is no string matching and no "best guess".

Range checks and the trip speed/distance check map to `UNIT_MISMATCH`. Nullability maps to `NULL_EXPLOSION`. Uniqueness maps to `DUPLICATE_KEYS`. Dtype and column presence map to `SCHEMA_DRIFT`. The PSI check names and `partition_arrival_sla` map to `DISTRIBUTION_SHIFT` and `LATE_PARTITION`.

## Consequences

- A column can move a lot inside its legal range and still fail only PSI, when the profile was fit on a different clean partition. The distribution-shift injector is written to do that.
- Fitting the profile from the generator's own bounds would make the detector a copy of the simulator. A held-out partition can disagree with a new seed, and that disagreement is a real false positive if it crosses 0.2. The threshold is not raised to hide it.
- A unit conversion that also exceeds a range check is still one fault type, because both checks are in the `UNIT_MISMATCH` row of the table.
- Adding a pandera check without a table entry fails closed at evaluation time.
- The threshold and the SLA are arguments, not constants buried in a check body. The defaults are 0.2 and 2 hours.

## Alternatives considered

- A learned anomaly model. Rejected because the benchmark needs a deterministic pass/fail and a named threshold.
- Kolmogorov-Smirnov against the same profile. Rejected in favor of PSI, which is the usual quantile-bin summary for data drift and is easy to explain next to a 0.2 cutoff.
- Building the bins with `linspace` on the generator's published interval. Rejected. Those bounds are not available for a real pipeline, and using them here would score the simulator against itself.
- Inferring the fault type from the check name with prefixes or keywords. Rejected. The table is the contract between the checks and `Incident.fault_type`.
