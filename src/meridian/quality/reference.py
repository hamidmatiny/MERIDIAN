"""Fit a quantile reference profile from a held-out frame."""

import numpy as np
import pandas as pd

from meridian.contracts.reference_profile import ColumnProfile, ReferenceProfile

DEFAULT_PSI_THRESHOLD = 0.2
_QUANTILE_BINS = 10


def fit_reference_profile(df: pd.DataFrame) -> ReferenceProfile:
    """Quantile bins for each numeric column, computed only from ``df``.

    Callers score a partition against a profile fit on a different clean
    partition. The edges are that frame's quantiles, not the generator's bounds.
    """
    dataset = df.attrs.get("dataset")
    if not isinstance(dataset, str) or dataset == "":
        raise ValueError("dataframe attrs must include a dataset name")
    columns = [
        profile
        for name in df.columns
        if (profile := _column_profile(str(name), df[name])) is not None
    ]
    if not columns:
        raise ValueError("dataframe has no numeric column to fit")
    return ReferenceProfile(
        dataset=dataset,
        psi_threshold=DEFAULT_PSI_THRESHOLD,
        columns=columns,
    )


def _column_profile(name: str, series: pd.Series) -> ColumnProfile | None:
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return None
    finite = series.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return None
    probabilities = np.linspace(0.0, 1.0, _QUANTILE_BINS + 1)
    edges = np.unique(np.quantile(finite, probabilities))
    if edges.size < 2:
        return None
    counts, _unused = np.histogram(finite, bins=edges)
    edges, counts = _drop_empty_bins(edges, counts)
    total = int(counts.sum())
    if edges.size < 2 or total == 0 or np.any(counts <= 0):
        return None
    return ColumnProfile(
        column=name,
        bin_edges=[float(edge) for edge in edges],
        expected_proportions=[float(count) / total for count in counts],
    )


def _drop_empty_bins(edges: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse quantile ties so every remaining bin has a positive count."""
    if not np.any(counts <= 0):
        return edges, counts
    kept = [float(edges[0])]
    kept_counts: list[int] = []
    pending = 0
    for count, right in zip(counts, edges[1:], strict=True):
        pending += int(count)
        if pending > 0:
            kept.append(float(right))
            kept_counts.append(pending)
            pending = 0
    return np.asarray(kept, dtype=float), np.asarray(kept_counts, dtype=int)
