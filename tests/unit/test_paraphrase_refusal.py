from dataclasses import FrozenInstanceError

import pytest

from cfprompt.paraphrase import (
    AdjustedParaphraseResult,
    is_refusal,
    select_best_undershoot,
)


@pytest.mark.unit
class TestIsRefusal:
    @pytest.mark.parametrize(
        "text",
        [
            "I can't help with that.",
            "I cannot assist with this request.",
            "I can't help you generate that.",
            "I cannot fulfill this request.",
            "I cannot fulfill the request.",
            "As an AI, I cannot...",
            "As a language model, I'm unable...",
            "I can't comply with the request.",
        ],
    )
    def test_known_refusal_phrases(self, text):
        assert is_refusal(text)

    def test_smart_quote_apostrophe_recognized(self):
        # U+2019 right single quotation mark
        assert is_refusal("I can’t help with that.")

    @pytest.mark.parametrize(
        "text",
        [
            "Sure, here is a paraphrase.",
            "The patient presents with...",
            "Here you go: [paraphrase]",
        ],
    )
    def test_non_refusals(self, text):
        assert not is_refusal(text)

    def test_case_insensitive(self):
        assert is_refusal("I CANNOT HELP WITH THIS")

    @pytest.mark.parametrize(
        "text",
        [
            "Sorry, I cannot help with that.",
            "I'm sorry, but I cannot fulfill this request.",
            "I’m sorry, but I can't comply.",
            "Unfortunately, I can't help with that.",
            "Unfortunately, I cannot do that.",
            "My apologies, I cannot answer.",
            "I am unable to help with that.",
            "I refuse to assist with this.",
        ],
    )
    def test_polite_preamble_refusals(self, text):
        """Refusals with polite preambles (Sorry / Unfortunately / My
        apologies / I am unable / I refuse) must be detected even when they
        are not a strict prefix of the response."""
        assert is_refusal(text)

    @pytest.mark.parametrize(
        "text",
        [
            "As an aid to nursing care, the nurse...",
            "As an aircraft was passing overhead...",
            "As an aide-de-camp, the officer...",
        ],
    )
    def test_as_an_ai_no_false_positive_on_aid(self, text):
        """The literal 'as an ai' prefix used to trigger on 'as an aid…'.
        With the bounded-prefix regex, the trigger requires the phrase to
        be followed by ',' or whitespace, not by an extra letter."""
        assert not is_refusal(text)

    def test_unicode_normalization_nfkc(self):
        """NFKC should fold a fullwidth-apostrophe variant or compatibility
        forms into ASCII before matching."""
        # U+02BC modifier letter apostrophe — used by some tokenizers.
        assert is_refusal("I canʼt help with that.")


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


@pytest.mark.unit
class TestSelectBestUndershoot:
    def test_picks_closest_undershoot(self):
        attempts = [
            {"paraphrase": "a", "actual_pct": 8.0, "deviation": 2.0},
            {"paraphrase": "b", "actual_pct": 9.5, "deviation": 0.5},
            {"paraphrase": "c", "actual_pct": 12.0, "deviation": 2.0},  # overshoot
        ]
        best = select_best_undershoot(attempts, target_pct=10.0)
        assert best["paraphrase"] == "b"

    def test_only_overshoots_falls_back_to_min_deviation(self):
        attempts = [
            {"paraphrase": "a", "actual_pct": 12.0, "deviation": 2.0},
            {"paraphrase": "b", "actual_pct": 11.0, "deviation": 1.0},
        ]
        best = select_best_undershoot(attempts, target_pct=10.0)
        assert best["paraphrase"] == "b"

    def test_at_target_treated_as_undershoot(self):
        attempts = [
            {"paraphrase": "a", "actual_pct": 10.0, "deviation": 0.0},
            {"paraphrase": "b", "actual_pct": 11.0, "deviation": 1.0},
        ]
        best = select_best_undershoot(attempts, target_pct=10.0)
        assert best["paraphrase"] == "a"
