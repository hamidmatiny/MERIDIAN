"""Partition arrival versus the SLA."""

from datetime import UTC, date, datetime, timedelta

from meridian.contracts.partition_metadata import PartitionMetadata

DEFAULT_FRESHNESS_SLA = timedelta(hours=2)


def partition_close(partition_date: date) -> datetime:
    """UTC midnight that ends ``partition_date``."""
    next_day = partition_date + timedelta(days=1)
    return datetime(next_day.year, next_day.month, next_day.day, tzinfo=UTC)


def deadline(partition_date: date, sla: timedelta = DEFAULT_FRESHNESS_SLA) -> datetime:
    """Latest on-time arrival for a partition."""
    return partition_close(partition_date) + sla


def on_time_arrival(
    partition_date: date,
    sla: timedelta = DEFAULT_FRESHNESS_SLA,
) -> datetime:
    """An arrival 30 minutes after close, inside the default two-hour SLA."""
    del sla
    return partition_close(partition_date) + timedelta(minutes=30)


def late_arrival(
    partition_date: date,
    intensity: float,
    sla: timedelta = DEFAULT_FRESHNESS_SLA,
) -> datetime:
    """An arrival strictly after the SLA.

    ``intensity`` scales the lateness up to 24 hours past the deadline.
    Every intensity in (0, 1] is late.
    """
    extra_s = max(1, int(intensity * 24 * 60 * 60))
    return deadline(partition_date, sla) + timedelta(seconds=extra_s)


def lateness_seconds(
    partition_date: date,
    intensity: float,
    sla: timedelta = DEFAULT_FRESHNESS_SLA,
) -> int:
    """Seconds past the deadline for :func:`late_arrival` at this intensity."""
    late = late_arrival(partition_date, intensity, sla)
    return int((late - deadline(partition_date, sla)).total_seconds())


def is_late(metadata: PartitionMetadata, sla: timedelta = DEFAULT_FRESHNESS_SLA) -> bool:
    """True when the partition arrived after its deadline."""
    return metadata.arrived_at > deadline(metadata.partition_date, sla)
