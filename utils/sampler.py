"""Weighted sampling to fight steering-label imbalance in CARLA data."""
from __future__ import annotations

import numpy as np


def steering_balanced_weights(
    steer: np.ndarray,
    num_bins: int = 21,
    smoothing: float = 0.05,
    max_weight: float = 100.0,
) -> np.ndarray:
    """Return a per-sample weight array inversely proportional to bin frequency.

    Steer is expected in [-1, 1]. We histogram |steer| into `num_bins` bins and
    weight each sample by 1 / (count + smoothing * total). The result is clipped
    to `max_weight` to prevent a handful of rare hard turns from dominating.
    """
    s = np.asarray(steer, dtype=np.float64).clip(-1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, num_bins + 1)
    # searchsorted -> integer bin in [1, num_bins]; subtract 1 and clamp.
    bins = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, num_bins - 1)
    counts = np.bincount(bins, minlength=num_bins).astype(np.float64)

    total = counts.sum()
    denom = counts + smoothing * total
    denom = np.where(denom <= 0, 1.0, denom)
    per_bin_w = 1.0 / denom

    # Normalize for readability (mean weight ~= 1 before clipping).
    per_bin_w = per_bin_w / per_bin_w[counts > 0].mean()
    per_bin_w = np.clip(per_bin_w, 0.0, max_weight)

    return per_bin_w[bins].astype(np.float32)
