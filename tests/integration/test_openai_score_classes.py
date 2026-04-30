# ABOUTME: Integration tests for OpenAIModel.score_classes top_logprobs path.
# ABOUTME: Uses canned API responses via monkeypatch on _request_one and _warmup.
import json
from pathlib import Path

import numpy as np
import pytest

from cfprompt.models.openai import OpenAIModel

FIX = Path(__file__).parent.parent / "fixtures" / "openai_responses"


def _load(name: str) -> dict:
    with (FIX / name).open() as f:
        return json.load(f)


def _make_model(monkeypatch):
    from cfprompt.models import openai as omod

    def fake_warmup(self):
        self._system_fingerprint = "fp_test_abc123"

    monkeypatch.setattr(omod.OpenAIModel, "_warmup", fake_warmup)
    return OpenAIModel(name="gpt-4.1", api_key="fake")


@pytest.mark.integration
class TestOpenAIScoreClasses:
    def test_extracts_normalized_class_probs(self, monkeypatch):
        m = _make_model(monkeypatch)
        canned = _load("score_classes_success.json")

        def fake_request(self, prompt, classes, seed):
            return canned

        monkeypatch.setattr(type(m), "_request_one", fake_request)
        probs = m.score_classes(["Q?"], ["A", "B", "C", "D"])
        assert probs.shape == (1, 4)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0], atol=1e-6)
        # P(A) should be the largest given logprobs in the fixture.
        assert probs[0].argmax() == 0

    def test_missing_class_returns_none_marker(self, monkeypatch):
        m = _make_model(monkeypatch)
        canned = _load("score_classes_missing_class.json")

        def fake_request(self, prompt, classes, seed):
            return canned

        monkeypatch.setattr(type(m), "_request_one", fake_request)
        probs = m.score_classes(["Q?"], ["Yes", "No"])
        # "No" is not in top_logprobs of the fixture; signal as NaN row.
        assert probs.shape == (1, 2)
        assert np.isnan(probs[0]).all()

    def test_columns_aligned_to_user_class_order(self, monkeypatch):
        m = _make_model(monkeypatch)
        canned = _load("score_classes_success.json")

        def fake_request(self, prompt, classes, seed):
            return canned

        monkeypatch.setattr(type(m), "_request_one", fake_request)
        probs1 = m.score_classes(["Q?"], ["A", "B", "C", "D"])
        probs2 = m.score_classes(["Q?"], ["D", "C", "B", "A"])
        np.testing.assert_allclose(probs1, probs2[:, ::-1], atol=1e-6)

    def test_low_prior_classes_still_scored(self, monkeypatch):
        """Even when len(classes)=2, the API should be asked for 20
        top_logprobs so that low-prior classes (buried below unrelated
        higher-probability tokens) still appear in the response."""
        m = _make_model(monkeypatch)
        canned = _load("score_classes_low_prior.json")

        captured: dict = {}

        def fake_create(self, prompt, classes, seed):
            captured["classes_len"] = len(classes)
            return canned

        monkeypatch.setattr(type(m), "_request_one", fake_create)
        probs = m.score_classes(["Q?"], ["Yes", "No"])
        assert probs.shape == (1, 2)
        # Both " Yes" and " No" appear (positions 18 and 19) only because the
        # request asked for 20 top_logprobs. With top_logprobs=2, neither
        # would be in the response and the row would be NaN.
        assert not np.isnan(probs[0]).any()
        np.testing.assert_allclose(probs[0].sum(), 1.0, atol=1e-6)

    def test_top_logprobs_request_is_always_20(self, monkeypatch):
        """Direct verification that _request_one passes top_logprobs=20
        regardless of len(classes)."""
        m = _make_model(monkeypatch)
        captured: dict = {}

        class FakeCompletions:
            def create(self_inner, **kwargs):
                captured.update(kwargs)

                class _Resp:
                    def model_dump(self_resp):
                        return {"choices": [{"logprobs": {"content": []}}]}

                return _Resp()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

            def __init__(self_inner, **kwargs):
                pass

        from cfprompt.models import openai as omod

        monkeypatch.setattr(omod, "OpenAI", FakeClient)
        m._request_one("p", ["A", "B"], seed=None)
        assert captured["top_logprobs"] == 20
