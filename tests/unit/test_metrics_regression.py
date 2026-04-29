import numpy as np
import pytest

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
