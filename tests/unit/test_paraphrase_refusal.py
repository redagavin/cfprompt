from dataclasses import FrozenInstanceError

import pytest

from cfprompt.paraphrase import AdjustedParaphraseResult, is_refusal


@pytest.mark.unit
class TestIsRefusal:
    @pytest.mark.parametrize("text", [
        "I can't help with that.",
        "I cannot assist with this request.",
        "I can't help you generate that.",
        "I cannot fulfill this request.",
        "I cannot fulfill the request.",
        "As an AI, I cannot...",
        "As a language model, I'm unable...",
        "I can't comply with the request.",
    ])
    def test_known_refusal_phrases(self, text):
        assert is_refusal(text)

    def test_smart_quote_apostrophe_recognized(self):
        # U+2019 right single quotation mark
        assert is_refusal("I can’t help with that.")

    @pytest.mark.parametrize("text", [
        "Sure, here is a paraphrase.",
        "The patient presents with...",
        "Here you go: [paraphrase]",
    ])
    def test_non_refusals(self, text):
        assert not is_refusal(text)

    def test_case_insensitive(self):
        assert is_refusal("I CANNOT HELP WITH THIS")


@pytest.mark.unit
class TestAdjustedParaphraseResult:
    def test_construct(self):
        r = AdjustedParaphraseResult(
            paraphrase="x",
            actual_edit_pct=10.0,
            retries_used=2,
            refused=False,
        )
        assert r.paraphrase == "x"
        assert r.actual_edit_pct == 10.0
        assert r.retries_used == 2
        assert r.refused is False

    def test_frozen(self):
        r = AdjustedParaphraseResult(
            paraphrase="x",
            actual_edit_pct=10.0,
            retries_used=0,
            refused=False,
        )
        with pytest.raises(FrozenInstanceError):
            r.paraphrase = "y"
