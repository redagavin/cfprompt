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
