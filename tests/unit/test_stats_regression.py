import numpy as np
import pytest

from cfprompt.metrics.regression import fit_difference
from cfprompt.report import TestResult
from cfprompt.stats import regression_test


@pytest.mark.unit
class TestRegressionTest:
    def test_one_sided_greater(self):
        rng = np.random.default_rng(0)
        direction = rng.choice([-1.0, 1.0], size=300)
        delta = 0.3 * direction + rng.normal(0, 0.1, size=300)
        fit = fit_difference(direction, delta)
        result = regression_test(fit, alternative="greater", n_distinct_samples=300)
        assert isinstance(result, TestResult)
        assert result.test == "ols_t"
        assert result.p_value_kind == "one-sided"
        assert result.p_value < 0.001
        assert result.p_value_two_sided is not None
        assert result.n == 300

    def test_one_sided_less_with_positive_signal_high_p(self):
        rng = np.random.default_rng(0)
        direction = rng.choice([-1.0, 1.0], size=300)
        delta = 0.3 * direction + rng.normal(0, 0.1, size=300)
        fit = fit_difference(direction, delta)
        result = regression_test(fit, alternative="less", n_distinct_samples=300)
        assert result.p_value > 0.95

    def test_extra_carries_beta_se(self):
        rng = np.random.default_rng(0)
        direction = rng.choice([-1.0, 1.0], size=100)
        delta = 0.5 * direction + rng.normal(0, 0.1, size=100)
        fit = fit_difference(direction, delta)
        result = regression_test(fit, alternative="greater", n_distinct_samples=100)
        assert result.extra["beta"] == fit.beta
        assert result.extra["se"] == fit.se
        assert result.extra["df_resid"] == fit.df_resid
