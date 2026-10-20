"""Build the stored quantile profile from the generating bounds."""

import numpy as np

from meridian.contracts.reference_profile import ColumnProfile, ReferenceProfile
from meridian.pipeline.catalog import DATASETS, DEFAULT_PSI_THRESHOLD, PSI_BINS, require_dataset


def reference_profile(
    dataset: str,
    *,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
) -> ReferenceProfile:
    """Quantile grid of the column the generator actually draws.

    The grid is fixed by the catalog bounds, not by a sampled file, so every
    seed is scored against the same profile.
    """
    spec = require_dataset(dataset)
    edges = np.linspace(spec.psi_min, spec.psi_max, PSI_BINS + 1).tolist()
    share = 1.0 / PSI_BINS
    column = ColumnProfile(
        column=spec.psi_column,
        bin_edges=edges,
        expected_proportions=[share] * PSI_BINS,
    )
    return ReferenceProfile(
        dataset=DATASETS[dataset].name,
        psi_threshold=psi_threshold,
        columns=[column],
    )
