import numpy as np
import pytest

from cfprompt.metrics.label import flip_rate
from cfprompt.report import TestResult
from cfprompt.stats import bootstrap_diff


@pytest.mark.unit
class TestBootstrapDiff:
    def test_returns_test_result(self):
        rng = np.random.default_rng(0)
        labels_orig = rng.choice(["A", "B"], size=100)
        labels_target = rng.choice(["A", "B"], size=100)
        labels_baseline = rng.choice(["A", "B"], size=100)
        result = bootstrap_diff(
            labels_orig=labels_orig,
            labels_target=labels_target,
            labels_baseline=labels_baseline,
            metric_fn=flip_rate,
            metric_name="flip_rate",
            n_bootstrap=200,
            seed=42,
        )
        assert isinstance(result, TestResult)
        assert result.test == "bootstrap"
        assert result.metric == "flip_rate"
        assert result.p_value_kind == "two-sided"
        assert result.ci_kind == "percentile"

    def test_recovers_signal_when_target_has_more_flips(self):
        rng = np.random.default_rng(0)
        n = 500
        labels_orig = rng.choice(["A", "B"], size=n)
        labels_target = np.array(
            [rng.choice(["A", "B"]) if rng.random() < 0.5 else label for label in labels_orig]
        )
        labels_baseline = np.array(
            [rng.choice(["A", "B"]) if rng.random() < 0.05 else label for label in labels_orig]
        )
        result = bootstrap_diff(
            labels_orig=labels_orig,
            labels_target=labels_target,
            labels_baseline=labels_baseline,
            metric_fn=flip_rate,
            metric_name="flip_rate",
            n_bootstrap=500,
            seed=42,
        )
        assert result.statistic > 0
        assert result.p_value < 0.05

    def test_paired_resampling(self):
        rng = np.random.default_rng(0)
        labels_orig = rng.choice([0, 1], size=100)
        labels_target = labels_orig.copy()
        labels_baseline = labels_orig.copy()
        result = bootstrap_diff(
            labels_orig=labels_orig,
            labels_target=labels_target,
            labels_baseline=labels_baseline,
            metric_fn=flip_rate,
            metric_name="flip_rate",
            n_bootstrap=100,
            seed=42,
        )
        assert result.statistic == 0.0
        assert result.extra.get("degenerate") is True
