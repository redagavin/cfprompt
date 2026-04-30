"""OLS regression for directional studies: difference and level models.

See spec §5.4 and §5.5.1 for the contract. Both models operate on
log p(outcome_class) (per-class log-probability), not binary log-odds
(BiB-style). v1 limitation acknowledged in the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import statsmodels.api as sm
from scipy import stats


@dataclass(frozen=True)
class RegressionFit:
    beta: float
    se: float
    t_stat: float
    p_value_two_sided: float
    ci_low_95: float
    ci_high_95: float
    n_obs: int
    df_resid: int
    r_squared: float
    extra: dict = field(default_factory=dict)


def _t_quantile_two_sided_975(df: int) -> float:
    return float(stats.t.ppf(0.975, df))


def fit_difference(direction: np.ndarray, delta: np.ndarray) -> RegressionFit:
    """Δᵢ = β · direction_i + ε_i.

    NO INTERCEPT. One row per sample. OLS classical (homoscedastic) SEs.
    `delta` is computed by stats.run_test as
        delta_i = log p(outcome | perturbed) - log p(outcome | paraphrase_baseline)
    before this function is called.
    """
    direction = np.asarray(direction, dtype=np.float64).reshape(-1, 1)
    delta = np.asarray(delta, dtype=np.float64)
    model = sm.OLS(delta, direction).fit()
    beta = float(model.params[0])
    se = float(model.bse[0])
    df_resid = int(model.df_resid)
    t_q = _t_quantile_two_sided_975(df_resid)
    return RegressionFit(
        beta=beta,
        se=se,
        t_stat=float(model.tvalues[0]),
        p_value_two_sided=float(model.pvalues[0]),
        ci_low_95=beta - t_q * se,
        ci_high_95=beta + t_q * se,
        n_obs=int(model.nobs),
        df_resid=df_resid,
        r_squared=float(model.rsquared),
        extra={},
    )


def fit_level(
    direction: np.ndarray,
    y: np.ndarray,
    z_i: np.ndarray,
    sample_id: np.ndarray,
) -> RegressionFit:
    """y = β₀ + β₁·z_i + β·direction + ε.

    Two stacked rows per sample (baseline row: y=log p(outcome|paraphrase_baseline),
    z_i=log p(outcome|original), direction=0; perturbed row: y=log p(outcome|perturbed),
    same z_i, direction=±1). Clustered SE on sample_id.
    """
    direction = np.asarray(direction, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z_i = np.asarray(z_i, dtype=np.float64)
    sample_id = np.asarray(sample_id)

    X = sm.add_constant(np.column_stack([z_i, direction]))
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample_id},
    )
    # Param order: const, z_i, direction
    beta_0 = float(model.params[0])
    beta_1 = float(model.params[1])
    beta_pert = float(model.params[2])
    se_pert = float(model.bse[2])
    df_resid = int(model.df_resid)
    # Use statsmodels' own conf_int so the CI quantile matches the SE/p-value
    # convention statsmodels chose for cluster-robust covariance (z-quantile
    # by default with use_t=False). Manually applying t_q(df_resid) on top of
    # cluster bse mixes conventions and produces a wider interval than the
    # underlying inference actually supports.
    ci_row = model.conf_int(alpha=0.05)
    ci_low_pert = float(np.asarray(ci_row)[2, 0])
    ci_high_pert = float(np.asarray(ci_row)[2, 1])
    return RegressionFit(
        beta=beta_pert,
        se=se_pert,
        t_stat=float(model.tvalues[2]),
        p_value_two_sided=float(model.pvalues[2]),
        ci_low_95=ci_low_pert,
        ci_high_95=ci_high_pert,
        n_obs=int(model.nobs),
        df_resid=df_resid,
        r_squared=float(model.rsquared),
        extra={
            "beta_0": beta_0,
            "beta_1": beta_1,
            "se_beta_0": float(model.bse[0]),
            "se_beta_1": float(model.bse[1]),
        },
    )
