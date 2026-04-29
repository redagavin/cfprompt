# ABOUTME: Integration tests for OpenAIModel construction, warmup, and cache_id.
# ABOUTME: Uses monkeypatch on _warmup so no network access is required.
import json
from pathlib import Path

import pytest

from cfprompt.models.openai import OpenAIModel

FIX = Path(__file__).parent.parent / "fixtures" / "openai_responses"


def _load(name: str) -> dict:
    with (FIX / name).open() as f:
        return json.load(f)


@pytest.mark.integration
class TestOpenAIModelInit:
    def test_warmup_resolves_fingerprint(self, monkeypatch):
        from cfprompt.models import openai as omod

        canned = _load("warmup_success.json")

        def fake_warmup(self):
            self._system_fingerprint = canned["system_fingerprint"]

        monkeypatch.setattr(omod.OpenAIModel, "_warmup", fake_warmup)
        m = OpenAIModel(name="gpt-4.1", api_key="fake-key")
        assert m.cache_id == (
            "openai:gpt-4.1|temp=0.0|top_p=1.0|max_tok=512"
            "|class_prefix=' '|fp=fp_test_abc123"
        )

    def test_warmup_missing_fingerprint_uses_uuid(self, monkeypatch):
        from cfprompt.models import openai as omod

        def fake_warmup(self):
            self._system_fingerprint = None

        monkeypatch.setattr(omod.OpenAIModel, "_warmup", fake_warmup)
        m = OpenAIModel(name="gpt-4.1", api_key="fake-key")
        assert "fp=unknown-" in m.cache_id

    def test_warmup_offline_uses_uuid(self, monkeypatch):
        from cfprompt.models import openai as omod

        def boom(self):
            raise OSError("offline")

        monkeypatch.setattr(omod.OpenAIModel, "_warmup", boom)
        m = OpenAIModel(name="gpt-4.1", api_key="fake-key")
        assert "fp=unknown-" in m.cache_id

    def test_temperature_top_p_max_tok_in_cache_id(self, monkeypatch):
        from cfprompt.models import openai as omod

        def fake_warmup(self):
            self._system_fingerprint = "fp_x"

        monkeypatch.setattr(omod.OpenAIModel, "_warmup", fake_warmup)
        m = OpenAIModel(
            name="gpt-4.1",
            api_key="fake-key",
            temperature=0.7,
            top_p=0.9,
            max_completion_tokens=128,
        )
        assert "temp=0.7" in m.cache_id
        assert "top_p=0.9" in m.cache_id
        assert "max_tok=128" in m.cache_id
