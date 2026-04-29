"""Aggregate label-based metrics: flip rate, mutual information, phi."""
from __future__ import annotations

import numpy as np

from cfprompt.exceptions import CfpromptError, DegenerateMetricError


def flip_rate(labels_orig: np.ndarray, labels_pert: np.ndarray) -> float:
    """Fraction of samples whose label changed."""
    labels_orig = np.asarray(labels_orig)
    labels_pert = np.asarray(labels_pert)
    if labels_orig.shape != labels_pert.shape:
        raise CfpromptError(
            f"flip_rate: shape mismatch {labels_orig.shape} vs {labels_pert.shape}"
        )
    if len(labels_orig) == 0:
        return 0.0
    return float((labels_orig != labels_pert).mean())


def mutual_information(labels_orig: np.ndarray, labels_pert: np.ndarray) -> float:
    """Mutual information of the joint label distribution.

    Computes MI on the empirical joint distribution of (orig, pert) labels.
    Non-negative. Symmetric. Returns 0 when labels are independent.
    """
    labels_orig = np.asarray(labels_orig)
    labels_pert = np.asarray(labels_pert)
    if labels_orig.shape != labels_pert.shape:
        raise CfpromptError(
            f"mutual_information: shape mismatch {labels_orig.shape} vs "
            f"{labels_pert.shape}"
        )
    n = len(labels_orig)
    if n == 0:
        return 0.0

    # Build joint table.
    pair_keys, pair_counts = np.unique(
        list(zip(labels_orig.tolist(), labels_pert.tolist(), strict=True)),
        axis=0,
        return_counts=True,
    )
    p_xy = pair_counts / n

    # Marginals.
    a_keys, a_counts = np.unique(labels_orig, return_counts=True)
    b_keys, b_counts = np.unique(labels_pert, return_counts=True)
    p_x = {k: c / n for k, c in zip(a_keys, a_counts, strict=True)}
    p_y = {k: c / n for k, c in zip(b_keys, b_counts, strict=True)}

    total = 0.0
    for (x, y), pxy in zip(pair_keys, p_xy, strict=True):
        if pxy == 0:
            continue
        denom = p_x[x] * p_y[y]
        total += pxy * (np.log(pxy) - np.log(denom))
    return float(total)


def phi_coefficient(labels_orig: np.ndarray, labels_pert: np.ndarray) -> float:
    """Phi coefficient on a 2x2 contingency table.

    Defined for binary labels only — raises CfpromptError on non-binary input.
    Raises DegenerateMetricError when a marginal of the 2x2 table is zero
    (caught by bootstrap and counted in extra['n_dropped_resamples']).
    """
    labels_orig = np.asarray(labels_orig)
    labels_pert = np.asarray(labels_pert)
    a_unique = np.unique(labels_orig)
    b_unique = np.unique(labels_pert)
    if len(a_unique) > 2 or len(b_unique) > 2:
        raise CfpromptError(
            f"phi_coefficient: defined for binary labels only; got "
            f"{a_unique.tolist()} / {b_unique.tolist()}"
        )

    # Use the union of label sets for the 2x2 table; pick the lexicographic
    # smaller as "0" and larger as "1" to canonicalize.
    union = sorted(set(a_unique.tolist()) | set(b_unique.tolist()))
    if len(union) > 2:
        raise CfpromptError(
            f"phi_coefficient: combined label set has more than 2 values: {union}"
        )
    if len(union) < 2:
        # Single-value labels => marginal zero on the other class.
        raise DegenerateMetricError(
            f"phi_coefficient: zero marginal — both arrays contain only label "
            f"{union[0]!r}"
        )
    lo, hi = union
    a01 = (labels_orig == hi).astype(int)
    b01 = (labels_pert == hi).astype(int)
    p11 = float(((a01 == 1) & (b01 == 1)).mean())
    p10 = float(((a01 == 1) & (b01 == 0)).mean())
    p01 = float(((a01 == 0) & (b01 == 1)).mean())
    p00 = float(((a01 == 0) & (b01 == 0)).mean())
    p1_ = p11 + p10
    p0_ = p01 + p00
    p_1 = p11 + p01
    p_0 = p10 + p00
    denom_sq = p1_ * p0_ * p_1 * p_0
    if denom_sq == 0:
        raise DegenerateMetricError(
            "phi_coefficient: zero marginal in 2x2 contingency table"
        )
    return float((p11 * p00 - p10 * p01) / np.sqrt(denom_sq))
