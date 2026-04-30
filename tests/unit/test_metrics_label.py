import numpy as np
import pytest

from cfprompt.exceptions import CfpromptError, DegenerateMetricError
from cfprompt.metrics.label import flip_rate, mutual_information, phi_coefficient


@pytest.mark.unit
class TestFlipRate:
    def test_no_flips(self):
        a = np.array(["A", "B", "A"])
        assert flip_rate(a, a) == 0.0

    def test_all_flips(self):
        a = np.array(["A", "A", "A"])
        b = np.array(["B", "B", "B"])
        assert flip_rate(a, b) == 1.0

    def test_partial(self):
        a = np.array(["A", "B", "A", "B"])
        b = np.array(["A", "A", "B", "B"])
        # Flips at indices 1 and 2 → 2/4 = 0.5
        assert flip_rate(a, b) == 0.5


@pytest.mark.unit
class TestMutualInformation:
    def test_perfect_dependence(self):
        # Identical labels → MI = entropy of the marginal
        a = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        v = mutual_information(a, a)
        assert v == pytest.approx(np.log(2), abs=1e-6)

    def test_independence_zero(self):
        # All combinations equally likely
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 1, 0, 1])
        v = mutual_information(a, b)
        assert v == pytest.approx(0.0, abs=1e-10)

    def test_mixed_dtype_labels_do_not_crash(self):
        """Regression: int vs str labels would KeyError on marginal lookup
        because np.unique(zip(...)) coerces to a 2D string array (R#4)."""
        a = np.array([0, 1])
        b = np.array(["A", "B"])
        v = mutual_information(a, b)
        # Two samples, perfect 1:1 mapping → MI = log 2.
        assert v == pytest.approx(np.log(2), abs=1e-6)

    def test_shape_mismatch_raises(self):
        a = np.array([0, 1, 0])
        b = np.array([0, 1])
        with pytest.raises(CfpromptError, match="shape mismatch"):
            mutual_information(a, b)

    def test_empty_input_returns_zero(self):
        a = np.array([], dtype=int)
        b = np.array([], dtype=int)
        assert mutual_information(a, b) == 0.0


@pytest.mark.unit
class TestPhi:
    def test_perfect_agreement(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 0, 1, 1])
        assert phi_coefficient(a, b) == pytest.approx(1.0)

    def test_perfect_disagreement(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([1, 1, 0, 0])
        assert phi_coefficient(a, b) == pytest.approx(-1.0)

    def test_zero_marginal_raises_degenerate(self):
        a = np.array([0, 0, 0, 0])
        b = np.array([0, 1, 0, 1])
        with pytest.raises(DegenerateMetricError, match="zero marginal"):
            phi_coefficient(a, b)

    def test_non_binary_raises_cfprompt_error(self):
        a = np.array([0, 1, 2, 0])
        b = np.array([0, 1, 1, 0])
        with pytest.raises(CfpromptError, match="binary"):
            phi_coefficient(a, b)
