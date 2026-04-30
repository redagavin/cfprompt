# ABOUTME: Integration tests for OpenAIModel construction, warmup, and cache_id.
# ABOUTME: Uses monkeypatch on _warmup so no network access is required.
import json
from pathlib import Path

import pytest

from cfprompt.models.openai import OpenAIModel, _safe_get_error_code

FIX = Path(__file__).parent.parent / "fixtures" / "openai_responses"


def _load(name: str) -> dict:
    with (FIX / name).open() as f:
        return json.load(f)


@pytest.mark.integration
class TestSafeGetErrorCode:
    def test_dict_body_returns_code(self):
        e = type("E", (), {"body": {"code": "rate_limit"}})()
        assert _safe_get_error_code(e) == "rate_limit"

    def test_dict_body_missing_code_returns_none(self):
        e = type("E", (), {"body": {"message": "oops"}})()
        assert _safe_get_error_code(e) is None

    def test_object_body_returns_code_attr(self):
        body = type("B", (), {"code": "context_length"})()
        e = type("E", (), {"body": body})()
        assert _safe_get_error_code(e) == "context_length"

    def test_string_body_returns_none(self):
        # The original lambda crashed on a string body.
        e = type("E", (), {"body": "raw string"})()
        assert _safe_get_error_code(e) is None

    def test_none_body_returns_none(self):
        e = type("E", (), {"body": None})()
        assert _safe_get_error_code(e) is None

    def test_no_body_attr_returns_none(self):
        e = ValueError("no body attribute at all")
        assert _safe_get_error_code(e) is None


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
            "openai:gpt-4.1|temp=0.0|top_p=1.0|max_tok=512|class_prefix=' '|fp=fp_test_abc123"
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

    def test_warmup_does_not_send_sampling_params(self, monkeypatch):
        """Reasoning models (o1, o3, gpt-5) reject temperature and top_p
        with a 400. Warmup should send only model + messages."""
        from cfprompt.models import openai as omod

        captured: dict = {}

        class FakeCompletions:
            def create(self_inner, **kwargs):
                captured.update(kwargs)

                class _Resp:
                    system_fingerprint = "fp_warmup_ok"

                return _Resp()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

            def __init__(self_inner, **kwargs):
                pass

        monkeypatch.setattr(omod, "OpenAI", FakeClient)
        m = OpenAIModel(
            name="o3",
            api_key="fake-key",
            temperature=0.7,
            top_p=0.9,
        )
        assert "temperature" not in captured
        assert "top_p" not in captured
        assert captured["model"] == "o3"
        assert m._system_fingerprint == "fp_warmup_ok"

    def test_warmup_propagates_400_on_unsupported_model(self, monkeypatch):
        """If the API still rejects warmup (e.g. unknown model), the offline
        UUID fallback engages. Simulate a 400 response and verify cache_id
        falls back to unknown- prefix."""
        from openai import APIStatusError

        from cfprompt.models import openai as omod

        class FakeCompletions:
            def create(self_inner, **kwargs):
                req = type("R", (), {"method": "POST", "url": "/chat"})()

                class _Resp:
                    status_code = 400
                    headers: dict = {}
                    request = req

                    def json(self_resp):
                        return {"error": {"message": "bad", "code": "bad"}}

                raise APIStatusError("bad", response=_Resp(), body={"code": "bad"})

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

            def __init__(self_inner, **kwargs):
                pass

        monkeypatch.setattr(omod, "OpenAI", FakeClient)
        m = OpenAIModel(name="bogus-model", api_key="fake-key")
        assert "fp=unknown-" in m.cache_id
