# ABOUTME: Integration tests for OpenAIModel.generate via chat completions.
# ABOUTME: Uses canned API responses via monkeypatch on _request_one_generate and _warmup.
import json
from pathlib import Path

import pytest

from cfprompt.models.openai import OpenAIModel

FIX = Path(__file__).parent.parent / "fixtures" / "openai_responses"


def _load(name: str) -> dict:
    with (FIX / name).open() as f:
        return json.load(f)


@pytest.mark.integration
class TestOpenAIGenerate:
    def test_returns_one_string_per_prompt(self, monkeypatch):
        from cfprompt.models import openai as omod

        def fake_warmup(self):
            self._system_fingerprint = "fp_x"

        canned = _load("generate_success.json")

        def fake_request(self, prompt, seed):
            return canned

        monkeypatch.setattr(omod.OpenAIModel, "_warmup", fake_warmup)
        monkeypatch.setattr(omod.OpenAIModel, "_request_one_generate", fake_request)
        m = OpenAIModel(name="gpt-4.1", api_key="fake")
        out = m.generate(["hi", "hello"])
        assert out == ["Hello, how can I help?", "Hello, how can I help?"]
