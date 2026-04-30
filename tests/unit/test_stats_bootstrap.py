import numpy as np
import pytest

from cfprompt.exceptions import DegenerateMetricError
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

    def test_bootstrap_unstable_when_many_resamples_drop(self):
        """When >5% of resamples raise DegenerateMetricError, the result
        must flag bootstrap_unstable=True so callers can downgrade trust
        in the p-value (R#4)."""
        rng = np.random.default_rng(0)
        n = 30
        labels_orig = rng.choice(["A", "B"], size=n)
        labels_target = rng.choice(["A", "B"], size=n)
        labels_baseline = rng.choice(["A", "B"], size=n)

        # Skip the first two calls (observed_target / observed_baseline) so the
        # observed metric is well-defined; thereafter raise on every 3rd call.
        call_counter = {"i": 0}

        def flaky_metric(orig, pert):
            call_counter["i"] += 1
            if call_counter["i"] <= 2:
                return float((orig != pert).mean())
            if call_counter["i"] % 3 == 0:
                raise DegenerateMetricError("simulated degenerate resample")
            return float((orig != pert).mean())

        n_bootstrap = 200
        result = bootstrap_diff(
            labels_orig=labels_orig,
            labels_target=labels_target,
            labels_baseline=labels_baseline,
            metric_fn=flaky_metric,
            metric_name="flip_rate",
            n_bootstrap=n_bootstrap,
            seed=42,
        )
        assert result.extra["bootstrap_unstable"] is True
        assert result.extra["n_dropped_resamples"] > 0.05 * n_bootstrap
