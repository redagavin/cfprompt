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
        assert result.n == 100
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert 0.0 <= result.p_value <= 1.0
        assert result.ci_low <= result.statistic <= result.ci_high

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

    def test_bootstrap_diff_matches_reference_script(self):
        """Bit-for-bit parity vs the paper's reference bootstrap script.

        cfprompt's bootstrap_diff must reproduce the published p-values
        exactly when given the same inputs. The reference function below is
        copied verbatim from scripts/compute_medqa_flip_rate_bootstrap.py
        (frozen for parity testing; do not edit unless the upstream changes).
        This guards against silent drift in the resampler.
        """

        def bootstrap_flip_rate_pvalue(gender_flips, benign_flips, n_bootstrap=10000, seed=42):
            rng = np.random.RandomState(seed)
            n = len(gender_flips)
            gender_flips = np.asarray(gender_flips, dtype=int)
            benign_flips = np.asarray(benign_flips, dtype=int)
            observed_diff = gender_flips.mean() - benign_flips.mean()
            boot_diffs = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                idx = rng.randint(0, n, size=n)
                boot_diffs[i] = gender_flips[idx].mean() - benign_flips[idx].mean()
            p_value = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
            p_value = min(p_value, 1.0)
            return {
                "gender_rate": gender_flips.mean(),
                "benign_rate": benign_flips.mean(),
                "observed_diff": observed_diff,
                "ci_low": np.percentile(boot_diffs, 2.5),
                "ci_high": np.percentile(boot_diffs, 97.5),
                "p_value": p_value,
            }

        rng = np.random.default_rng(0)
        n = 200
        labels_orig = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
        labels_target = labels_orig.copy()
        flip_mask = rng.random(n) < 0.20
        labels_target[flip_mask] = np.where(labels_target[flip_mask] == "A", "B", "A")
        labels_baseline = labels_orig.copy()
        flip_mask2 = rng.random(n) < 0.05
        labels_baseline[flip_mask2] = np.where(labels_baseline[flip_mask2] == "A", "B", "A")

        result = bootstrap_diff(
            labels_orig=labels_orig,
            labels_target=labels_target,
            labels_baseline=labels_baseline,
            metric_fn=flip_rate,
            metric_name="flip_rate",
            n_bootstrap=10000,
            seed=42,
        )

        gender_flips = (labels_orig != labels_target).astype(int)
        benign_flips = (labels_orig != labels_baseline).astype(int)
        reference = bootstrap_flip_rate_pvalue(
            gender_flips, benign_flips, n_bootstrap=10000, seed=42
        )

        assert result.statistic == pytest.approx(reference["observed_diff"], abs=1e-12)
        assert result.p_value == pytest.approx(reference["p_value"], abs=1e-12)
        assert result.ci_low == pytest.approx(reference["ci_low"], abs=1e-12)
        assert result.ci_high == pytest.approx(reference["ci_high"], abs=1e-12)

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
