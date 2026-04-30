import numpy as np
import pytest
import statsmodels.api as sm

from cfprompt.metrics.regression import RegressionFit, fit_difference, fit_level


@pytest.mark.unit
class TestFitDifference:
    def test_recovers_known_beta(self):
        rng = np.random.default_rng(0)
        n = 1000
        direction = rng.choice([-1.0, 1.0], size=n)
        true_beta = 0.5
        delta = true_beta * direction + rng.normal(0, 0.1, size=n)
        fit = fit_difference(direction, delta)
        assert isinstance(fit, RegressionFit)
        assert fit.beta == pytest.approx(true_beta, abs=0.05)
        assert fit.n_obs == n
        # No intercept => 1 parameter
        assert fit.df_resid == n - 1

    def test_two_sided_p_small_for_clear_signal(self):
        rng = np.random.default_rng(0)
        n = 200
        direction = rng.choice([-1.0, 1.0], size=n)
        delta = 0.3 * direction + rng.normal(0, 0.1, size=n)
        fit = fit_difference(direction, delta)
        assert fit.p_value_two_sided < 0.001

    def test_ci_brackets_beta(self):
        rng = np.random.default_rng(0)
        direction = rng.choice([-1.0, 1.0], size=500)
        delta = 0.5 * direction + rng.normal(0, 0.1, size=500)
        fit = fit_difference(direction, delta)
        assert fit.ci_low_95 < fit.beta < fit.ci_high_95


@pytest.mark.unit
class TestFitLevel:
    def test_recovers_known_beta_pert(self):
        rng = np.random.default_rng(1)
        g = 200
        direction_per_sample = rng.choice([-1.0, 1.0], size=g)
        z = rng.normal(0, 1, size=g)
        # Ground truth: y_baseline = z + noise; y_perturbed = z + 0.3*direction + noise.
        y_base = z + rng.normal(0, 0.05, size=g)
        y_pert = z + 0.3 * direction_per_sample + rng.normal(0, 0.05, size=g)
        # Stack: 2g rows.
        direction_stacked = np.concatenate([np.zeros(g), direction_per_sample])
        y_stacked = np.concatenate([y_base, y_pert])
        z_stacked = np.concatenate([z, z])
        sample_id = np.concatenate([np.arange(g), np.arange(g)])
        fit = fit_level(direction_stacked, y_stacked, z_stacked, sample_id)
        assert fit.beta == pytest.approx(0.3, abs=0.05)
        assert fit.n_obs == 2 * g
        # 3 parameters: intercept, z_i, direction
        assert fit.df_resid == 2 * g - 3
        # Level model exposes beta_0 and beta_1 in extra
        assert "beta_0" in fit.extra
        assert "beta_1" in fit.extra
        assert "se_beta_0" in fit.extra
        assert "se_beta_1" in fit.extra

    def test_ci_matches_statsmodels_conf_int(self):
        """fit_level CI must match statsmodels' own conf_int for the perturbed-
        direction coefficient — manually applying a t-quantile on top of the
        cluster-robust SE produced a CI inconsistent with statsmodels' p-value
        convention (R#4)."""
        rng = np.random.default_rng(2)
        g = 80
        direction_per_sample = rng.choice([-1.0, 1.0], size=g)
        z = rng.normal(0, 1, size=g)
        y_base = z + rng.normal(0, 0.05, size=g)
        y_pert = z + 0.3 * direction_per_sample + rng.normal(0, 0.05, size=g)
        direction_stacked = np.concatenate([np.zeros(g), direction_per_sample])
        y_stacked = np.concatenate([y_base, y_pert])
        z_stacked = np.concatenate([z, z])
        sample_id = np.concatenate([np.arange(g), np.arange(g)])
        fit = fit_level(direction_stacked, y_stacked, z_stacked, sample_id)
        # Recompute via statsmodels and compare CI.
        X = sm.add_constant(np.column_stack([z_stacked, direction_stacked]))
        model = sm.OLS(y_stacked, X).fit(cov_type="cluster", cov_kwds={"groups": sample_id})
        ci = np.asarray(model.conf_int(alpha=0.05))[2]
        assert fit.ci_low_95 == pytest.approx(float(ci[0]), abs=1e-10)
        assert fit.ci_high_95 == pytest.approx(float(ci[1]), abs=1e-10)

    def test_clustered_se_exceeds_naive_with_within_cluster_correlation(self):
        """With strong within-cluster correlation in y, clustered SE > naive SE
        for the perturbed-direction coefficient. Sanity-checks that the
        cov_type='cluster' option is actually being applied (R#4)."""
        rng = np.random.default_rng(3)
        g = 60
        direction_per_sample = rng.choice([-1.0, 1.0], size=g)
        # Per-cluster intercept shock — induces very high within-cluster corr.
        cluster_shock = rng.normal(0, 1.0, size=g)
        z = rng.normal(0, 0.1, size=g)
        eps_pert = rng.normal(0, 0.05, size=g)
        eps_base = rng.normal(0, 0.05, size=g)
        y_base = cluster_shock + z + eps_base
        y_pert = cluster_shock + z + 0.3 * direction_per_sample + eps_pert
        direction_stacked = np.concatenate([np.zeros(g), direction_per_sample])
        y_stacked = np.concatenate([y_base, y_pert])
        z_stacked = np.concatenate([z, z])
        sample_id = np.concatenate([np.arange(g), np.arange(g)])
        fit = fit_level(direction_stacked, y_stacked, z_stacked, sample_id)
        # Naive (homoscedastic) SE for comparison.
        X = sm.add_constant(np.column_stack([z_stacked, direction_stacked]))
        naive = sm.OLS(y_stacked, X).fit()
        assert fit.se > float(naive.bse[2])
