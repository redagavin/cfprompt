"""Statistical tests with metric-type dispatch.

See spec §5.5 for the auto-pick rule:
- Per-sample (jsd, kl) → paired t-test (two-sided)
- Aggregate (flip_rate, mi, phi) → bootstrap (two-sided, paper formula)
- Regression → OLS t-test on β (one-sided per `alternative`)
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as scipy_stats

from .report import TestResult


def paired_t(
    target: np.ndarray,
    baseline: np.ndarray,
    metric_name: str,
) -> TestResult:
    """Paired t-test on Δ_i = target_i - baseline_i. Two-sided.

    Convention: t-statistic's sign reflects mean(target) - mean(baseline).
    """
    target = np.asarray(target, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if target.shape != baseline.shape:
        raise ValueError(f"shape mismatch: {target.shape} vs {baseline.shape}")
    n = len(target)
    diffs = target - baseline
    mean_diff = float(diffs.mean()) if n else 0.0
    std_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0

    if n < 2 or std_diff == 0.0:
        return TestResult(
            metric=metric_name,
            test="paired_t",
            statistic=0.0,
            p_value=1.0,
            p_value_kind="two-sided",
            ci_low=0.0,
            ci_high=0.0,
            ci_kind=None,
            n=n,
            extra={
                "degenerate": True,
                "reason": "zero_variance",
                "mean_diff": mean_diff,
                "std_diff": std_diff,
            },
        )
    res = scipy_stats.ttest_rel(target, baseline, alternative="two-sided")
    se = std_diff / math.sqrt(n)
    df = n - 1
    t_q = float(scipy_stats.t.ppf(0.975, df))
    return TestResult(
        metric=metric_name,
        test="paired_t",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        p_value_kind="two-sided",
        ci_low=mean_diff - t_q * se,
        ci_high=mean_diff + t_q * se,
        ci_kind="asymptotic",
        n=n,
        extra={
            "degenerate": False,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
        },
    )
