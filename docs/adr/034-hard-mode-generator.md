# ADR-034: Hard mode is clean data with realistic noise

## Context

A detector that only sees injected faults can look perfect while alarming on ordinary
partitions. The bench needs a false-positive rate on data that is still legal. The
requested "ADR-032" for this generator is this document, because ADR-032 is the recorded
timeline.

## Decision

Hard mode starts from the clean generator and does not call an injector. Numeric columns
get a small gaussian nudge and a one-percent Laplace tail, then clip to the pandera
bounds. Two rows sit exactly on the bounds. Category ids on telemetry and trips are drawn
from a Zipf so many ids appear once and none are null. Fleet ids stay unique because they
are the primary key. Trip speed is rewritten from distance and duration so the physical
check still holds. The partition arrives one minute before the SLA deadline: later than
the usual on-time arrival, and not late. Sentinel incidents on these partitions are false
positives. The rate is runs that emitted an incident, over hard runs, plus the incident
count.

## Consequences

A hard run that trips PSI or a range check is a false positive, not a bug in the
generator to be tuned away after the number is seen. The generator does not insert nulls
or duplicate keys.

## Alternatives

Use intensity 0 of an injector. Rejected: that is the clean generator with no noise, so
it does not exercise bounds, tails, or a late-but-legal arrival. Inject faults and expect
zero detections. Rejected: that measures recall, not the false-positive rate.
