"""Population Stability Index against a stored quantile profile."""

import math

import numpy as np

from meridian.contracts.reference_profile import ColumnProfile

_EPSILON = 1e-6


def population_stability_index(values: np.ndarray, profile: ColumnProfile) -> float:
    """Compare ``values`` to ``profile`` using the classic PSI sum.

    Values outside the stored edges are left out of the histogram. A unit
    conversion that leaves the generating range therefore does not look like
    a distribution shift. An empty sample returns infinity.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.inf
    counts, _edges = np.histogram(finite, bins=np.asarray(profile.bin_edges, dtype=float))
    total = int(counts.sum())
    if total == 0:
        return math.inf
    observed = counts / total
    expected = np.asarray(profile.expected_proportions, dtype=float)
    observed = np.clip(observed, _EPSILON, None)
    expected = np.clip(expected, _EPSILON, None)
    observed = observed / observed.sum()
    expected = expected / expected.sum()
    return float(np.sum((observed - expected) * np.log(observed / expected)))
