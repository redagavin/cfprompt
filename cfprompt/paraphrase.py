"""Adjusted paraphrase baseline generator (port of calibrated_paraphrase.py)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

_logger = logging.getLogger("cfprompt")


REFUSAL_PHRASES = [
    "i can't help with",
    "i cannot help with",
    "i can't help you",
    "i cannot help you",
    "i can't help engineer",
    "i cannot help engineer",
    "i can't help generate",
    "i cannot help generate",
    "i can't assist",
    "i cannot assist",
    "i can't do what",
    "i cannot do what",
    "i can't do that",
    "i cannot do that",
    "i cannot fulfill this",
    "i can't fulfill this",
    "i cannot fulfill the",
    "i can't fulfill the",
    "i cannot fulfill your",
    "i can't fulfill your",
    "i can't comply with the",
    "i cannot comply with the",
    "i can't comply with request",
    "i cannot comply with request",
    "i can't produce a paraphrase",
    "i cannot produce a paraphrase",
    "i can't meet the requirement",
    "i cannot meet the requirement",
    "i can't generate a paraphrase",
    "i cannot generate a paraphrase",
    "as an ai",
    "as a language model",
]


def is_refusal(text: str) -> bool:
    """Detect refusal-ish responses by phrase prefix match.

    Case-insensitive; smart-quote apostrophe (U+2019) is normalized.
    """
    normalized = text.strip().lower().replace("’", "'")
    return any(normalized.startswith(p) for p in REFUSAL_PHRASES)


@dataclass(frozen=True)
class AdjustedParaphraseResult:
    """Outcome of one paraphrase request (after retries)."""

    paraphrase: str            # the chosen paraphrase text (or original on full refusal)
    actual_edit_pct: float     # achieved token edit %; 0.0 when refused
    retries_used: int          # number of retries that produced any paraphrase
    refused: bool              # True iff every attempt was a refusal


def select_best_undershoot(attempts: list[dict], target_pct: float) -> dict:
    """From a list of attempt dicts (each with 'actual_pct' and 'deviation'),
    return the one with smallest deviation among those whose actual_pct
    does NOT exceed `target_pct`. If every attempt overshoots, return the
    one with smallest deviation overall.
    """
    undershoots = [a for a in attempts if a["actual_pct"] <= target_pct]
    if not undershoots:
        return min(attempts, key=lambda x: x["deviation"])
    return min(undershoots, key=lambda x: x["deviation"])
