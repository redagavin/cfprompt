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

    def test_max_concurrent_kwarg_rejected(self, monkeypatch):
        """max_concurrent was documented as v2 future work and never used.
        Removing it should raise TypeError if any caller still passes it."""
        from cfprompt.models import openai as omod

        def fake_warmup(self):
            self._system_fingerprint = "fp_x"

        monkeypatch.setattr(omod.OpenAIModel, "_warmup", fake_warmup)
        with pytest.raises(TypeError):
            OpenAIModel(name="gpt-4.1", api_key="fake", max_concurrent=4)


@pytest.mark.integration
class TestOpenAIFingerprintDrift:
    @staticmethod
    def _patch(monkeypatch, fingerprints: list[str]):
        """Build an OpenAIModel whose chat.completions.create yields the
        given fingerprints in order."""
        from cfprompt.models import openai as omod

        # Warmup uses the OpenAI client too; resolve it to fingerprints[0].
        i = {"n": 0}

        class FakeCompletions:
            def create(self_inner, **kwargs):
                idx = i["n"]
                i["n"] += 1
                fp = fingerprints[min(idx, len(fingerprints) - 1)]

                class _Resp:
                    system_fingerprint = fp

                    def model_dump(self_resp):
                        return {
                            "system_fingerprint": fp,
                            "choices": [
                                {
                                    "message": {"content": "ok"},
                                    "logprobs": {
                                        "content": [
                                            {
                                                "top_logprobs": [
                                                    {"token": " A", "logprob": -0.1},
                                                ]
                                            }
                                        ]
                                    },
                                }
                            ],
                        }

                return _Resp()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

            def __init__(self_inner, **kwargs):
                pass

            def close(self_inner):
                pass

        monkeypatch.setattr(omod, "OpenAI", FakeClient)
        return OpenAIModel(name="gpt-4.1", api_key="fake-key")

    def test_drift_warns_once_per_new_fingerprint(self, monkeypatch, caplog):
        # warmup -> fp_a; then 3 score_classes calls returning fp_b, fp_b, fp_c.
        m = self._patch(monkeypatch, ["fp_a", "fp_b", "fp_b", "fp_c"])
        with caplog.at_level("WARNING", logger="cfprompt"):
            m.score_classes(["q"], ["A"])  # observes fp_b — warn
            m.score_classes(["q"], ["A"])  # observes fp_b again — silent
            m.score_classes(["q"], ["A"])  # observes fp_c — warn
        drift_msgs = [r for r in caplog.records if "system_fingerprint drift" in r.message]
        assert len(drift_msgs) == 2
        assert "fp_b" in drift_msgs[0].message
        assert "fp_c" in drift_msgs[1].message

    def test_no_drift_no_warning(self, monkeypatch, caplog):
        m = self._patch(monkeypatch, ["fp_a", "fp_a"])
        with caplog.at_level("WARNING", logger="cfprompt"):
            m.score_classes(["q"], ["A"])
        drift_msgs = [r for r in caplog.records if "system_fingerprint drift" in r.message]
        assert drift_msgs == []


@pytest.mark.integration
class TestOpenAIClose:
    @staticmethod
    def _make(monkeypatch):
        from cfprompt.models import openai as omod

        closed = {"called": 0}

        class FakeCompletions:
            def create(self_inner, **kwargs):
                class _Resp:
                    system_fingerprint = "fp_x"

                    def model_dump(self_resp):
                        return {
                            "system_fingerprint": "fp_x",
                            "choices": [
                                {
                                    "message": {"content": "ok"},
                                    "logprobs": {
                                        "content": [
                                            {
                                                "top_logprobs": [
                                                    {"token": " A", "logprob": -0.1},
                                                ]
                                            }
                                        ]
                                    },
                                }
                            ],
                        }

                return _Resp()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

            def __init__(self_inner, **kwargs):
                pass

            def close(self_inner):
                closed["called"] += 1

        monkeypatch.setattr(omod, "OpenAI", FakeClient)
        m = OpenAIModel(name="gpt-4.1", api_key="fake-key")
        return m, closed

    def test_close_idempotent(self, monkeypatch):
        m, closed = self._make(monkeypatch)
        # Trigger client creation by issuing one request.
        m.score_classes(["q"], ["A"])
        m.close()
        m.close()  # second call is a no-op
        assert closed["called"] == 1

    def test_request_after_close_raises(self, monkeypatch):
        m, _ = self._make(monkeypatch)
        m.score_classes(["q"], ["A"])
        m.close()
        with pytest.raises(RuntimeError, match="model closed"):
            m.score_classes(["q"], ["A"])
        with pytest.raises(RuntimeError, match="model closed"):
            m.generate(["q"])

    def test_close_without_request_is_safe(self, monkeypatch):
        # The cached client is created lazily on the first scoring/generate
        # request. Without one, close() is a pure flag flip.
        m, closed = self._make(monkeypatch)
        m.close()
        assert closed["called"] == 0
        with pytest.raises(RuntimeError, match="model closed"):
            m.score_classes(["q"], ["A"])
