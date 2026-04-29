import numpy as np
import pytest

from cfprompt.metrics.distributional import jsd, kl


@pytest.mark.unit
class TestJSD:
    def test_identity_zero(self):
        p = np.array([[0.5, 0.5], [0.25, 0.75]])
        out = jsd(p, p)
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-10)

    def test_symmetric(self):
        p = np.array([[0.7, 0.3]])
        q = np.array([[0.2, 0.8]])
        np.testing.assert_allclose(jsd(p, q), jsd(q, p), atol=1e-10)

    def test_known_value_uniform_vs_dirac(self):
        # JSD between uniform [0.5,0.5] and one-hot [1,0]:
        # = 0.5 * KL([1,0] || M) + 0.5 * KL([0.5,0.5] || M), where M=[0.75,0.25]
        # = 0.5 * (1 * log(1/0.75)) + 0.5 * (0.5*log(0.5/0.75) + 0.5*log(0.5/0.25))
        p = np.array([[1.0, 0.0]])
        q = np.array([[0.5, 0.5]])
        m = 0.5 * (p + q)
        # KL(p||m) where p has zero entries: 0*log(0) := 0
        kl_p_m = (p * (np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(m, 1e-12, 1)))).sum(axis=1)
        kl_q_m = (q * (np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(m, 1e-12, 1)))).sum(axis=1)
        expected = 0.5 * kl_p_m + 0.5 * kl_q_m
        np.testing.assert_allclose(jsd(p, q), expected, atol=1e-10)


@pytest.mark.unit
class TestKL:
    def test_identity_zero(self):
        p = np.array([[0.6, 0.4], [0.1, 0.9]])
        np.testing.assert_allclose(kl(p, p), [0.0, 0.0], atol=1e-10)

    def test_known_value(self):
        # KL([0.5,0.5] || [0.25,0.75]) = 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75)
        p = np.array([[0.5, 0.5]])
        q = np.array([[0.25, 0.75]])
        expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
        np.testing.assert_allclose(kl(p, q), [expected], atol=1e-10)

    def test_clip_handles_zero_q(self):
        # Without clipping, KL([0.5,0.5] || [1.0,0.0]) is +inf. With eps clip,
        # finite (large) value.
        p = np.array([[0.5, 0.5]])
        q = np.array([[1.0, 0.0]])
        v = kl(p, q)
        assert np.isfinite(v).all()
        assert v[0] > 1.0  # Big but finite.

    def test_clip_handles_zero_p(self):
        p = np.array([[1.0, 0.0]])
        q = np.array([[0.5, 0.5]])
        v = kl(p, q)
        # KL(P||Q) with P having 0 entry: 0*log(0/q) := 0 contribution.
        # So result = 1 * log(1/0.5) = log 2.
        np.testing.assert_allclose(v, [np.log(2)], atol=1e-6)
