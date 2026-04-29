"""Statistical tests with metric-type dispatch.

See spec §5.5 for the auto-pick rule:
- Per-sample (jsd, kl) → paired t-test (two-sided)
- Aggregate (flip_rate, mi, phi) → bootstrap (two-sided, paper formula)
- Regression → OLS t-test on β (one-sided per `alternative`)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy import stats as scipy_stats

from .exceptions import DegenerateMetricError
from .metrics.regression import RegressionFit
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


def bootstrap_diff(
    labels_orig: np.ndarray,
    labels_target: np.ndarray,
    labels_baseline: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> TestResult:
    """Paired bootstrap on metric(target) - metric(baseline).

    See spec §5.5 for the formula:
        p_value = min(1.0, 2.0 * min(P(diff <= 0), P(diff >= 0)))
    matching scripts/compute_medqa_flip_rate_bootstrap.py.
    """
    labels_orig = np.asarray(labels_orig)
    labels_target = np.asarray(labels_target)
    labels_baseline = np.asarray(labels_baseline)
    n = len(labels_orig)

    rng = np.random.default_rng(seed)

    try:
        observed_target = metric_fn(labels_orig, labels_target)
        observed_baseline = metric_fn(labels_orig, labels_baseline)
    except DegenerateMetricError as e:
        return TestResult(
            metric=metric_name,
            test="bootstrap",
            statistic=None,
            p_value=1.0,
            p_value_kind="two-sided",
            ci_low=None,
            ci_high=None,
            ci_kind=None,
            n=n,
            extra={
                "degenerate": True,
                "reason": f"observed_metric_undefined: {e}",
                "n_resamples": n_bootstrap,
                "n_used_resamples": 0,
                "n_dropped_resamples": 0,
                "bootstrap_unstable": False,
                "observed_delta": None,
            },
        )
    observed_delta = observed_target - observed_baseline

    diffs: list[float] = []
    n_dropped = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            t = metric_fn(labels_orig[idx], labels_target[idx])
            b = metric_fn(labels_orig[idx], labels_baseline[idx])
        except (DegenerateMetricError, ZeroDivisionError):
            n_dropped += 1
            continue
        diffs.append(t - b)

    n_used = len(diffs)
    if n_used == 0:
        return TestResult(
            metric=metric_name,
            test="bootstrap",
            statistic=observed_delta,
            p_value=1.0,
            p_value_kind="two-sided",
            ci_low=observed_delta,
            ci_high=observed_delta,
            ci_kind="percentile",
            n=n,
            extra={
                "degenerate": True,
                "reason": "all_resamples_dropped",
                "n_resamples": n_bootstrap,
                "n_used_resamples": 0,
                "n_dropped_resamples": n_bootstrap,
                "bootstrap_unstable": False,
                "observed_delta": observed_delta,
            },
        )

    diffs_arr = np.asarray(diffs, dtype=np.float64)
    p_le = float(np.mean(diffs_arr <= 0))
    p_ge = float(np.mean(diffs_arr >= 0))
    p_value = min(1.0, 2.0 * min(p_le, p_ge))

    if np.allclose(diffs_arr, diffs_arr[0]):
        return TestResult(
            metric=metric_name,
            test="bootstrap",
            statistic=observed_delta,
            p_value=1.0,
            p_value_kind="two-sided",
            ci_low=observed_delta,
            ci_high=observed_delta,
            ci_kind="percentile",
            n=n,
            extra={
                "degenerate": True,
                "reason": "zero_variance",
                "n_resamples": n_bootstrap,
                "n_used_resamples": n_used,
                "n_dropped_resamples": n_dropped,
                "bootstrap_unstable": False,
                "observed_delta": observed_delta,
            },
        )

    ci_low, ci_high = np.percentile(diffs_arr, [2.5, 97.5])
    bootstrap_unstable = n_dropped > 0.05 * n_bootstrap
    return TestResult(
        metric=metric_name,
        test="bootstrap",
        statistic=observed_delta,
        p_value=p_value,
        p_value_kind="two-sided",
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_kind="percentile",
        n=n,
        extra={
            "degenerate": False,
            "n_resamples": n_bootstrap,
            "n_used_resamples": n_used,
            "n_dropped_resamples": n_dropped,
            "bootstrap_unstable": bootstrap_unstable,
            "observed_delta": observed_delta,
        },
    )


def regression_test(
    fit: RegressionFit,
    alternative: Literal["greater", "less"],
    n_distinct_samples: int,
) -> TestResult:
    """OLS t-test on β; one-sided per `alternative`.

    p-value derivation:
      alternative == "less":    p = scipy.stats.t.cdf(t_stat, df_resid)
      alternative == "greater": p = scipy.stats.t.sf(t_stat, df_resid)

    `n_distinct_samples` is set as TestResult.n; for the level model this is
    NOT fit.n_obs (which is 2 * samples for stacked rows).
    """
    if alternative == "less":
        p_one = float(scipy_stats.t.cdf(fit.t_stat, fit.df_resid))
    elif alternative == "greater":
        p_one = float(scipy_stats.t.sf(fit.t_stat, fit.df_resid))
    else:
        raise ValueError(f"alternative must be 'greater' or 'less'; got {alternative!r}")

    extra = {
        "beta": fit.beta,
        "se": fit.se,
        "df_resid": fit.df_resid,
        "r_squared": fit.r_squared,
        "regression_model": "level" if "beta_0" in fit.extra else "difference",
        "degenerate": False,
    }
    extra.update(fit.extra)

    return TestResult(
        metric="regression",
        test="ols_t",
        statistic=fit.t_stat,
        p_value=p_one,
        p_value_kind="one-sided",
        p_value_two_sided=fit.p_value_two_sided,
        ci_low=fit.ci_low_95,
        ci_high=fit.ci_high_95,
        ci_kind="asymptotic",
        n=n_distinct_samples,
        extra=extra,
    )
