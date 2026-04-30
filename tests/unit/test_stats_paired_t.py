import numpy as np
import pytest

from cfprompt.report import TestResult
from cfprompt.stats import paired_t


@pytest.mark.unit
class TestPairedT:
    def test_returns_test_result(self):
        target = np.array([0.5, 0.6, 0.7])
        baseline = np.array([0.1, 0.2, 0.3])
        result = paired_t(target, baseline, metric_name="jsd")
        assert isinstance(result, TestResult)
        assert result.test == "paired_t"
        assert result.metric == "jsd"
        assert result.p_value_kind == "two-sided"
        assert result.n == 3

    def test_recovers_known_signal(self):
        rng = np.random.default_rng(0)
        n = 200
        target = rng.normal(loc=0.5, scale=0.1, size=n)
        baseline = rng.normal(loc=0.0, scale=0.1, size=n)
        result = paired_t(target, baseline, metric_name="jsd")
        assert result.statistic > 0
        assert result.p_value < 0.001

    def test_no_signal_high_p(self):
        rng = np.random.default_rng(1)
        target = rng.normal(loc=0.0, scale=0.1, size=200)
        baseline = rng.normal(loc=0.0, scale=0.1, size=200)
        result = paired_t(target, baseline, metric_name="jsd")
        assert result.p_value > 0.05

    def test_zero_variance_returns_degenerate(self):
        target = np.array([0.5, 0.5, 0.5])
        baseline = np.array([0.5, 0.5, 0.5])
        result = paired_t(target, baseline, metric_name="jsd")
        assert result.p_value == 1.0
        assert result.extra.get("degenerate") is True

    def test_shape_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            paired_t(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), metric_name="jsd")
