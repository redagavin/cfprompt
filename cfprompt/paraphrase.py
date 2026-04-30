"""Adjusted paraphrase baseline generator (port of calibrated_paraphrase.py)."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

from .tokenization import token_edit_distance_pct

_logger = logging.getLogger("cfprompt")


# Strong-prefix patterns: only match when the (normalized) text starts with
# one of these phrases. Used for clear-cut "I can't…" / "I cannot…" leads.
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
]


# "As an AI" / "As a language model" need stricter boundaries: bare prefix
# "as an ai" matches "as an aid…" (false positive). Require the phrase to be
# followed by punctuation, whitespace, or end-of-text.
_REFUSAL_PREFIX_REGEXES = [
    re.compile(r"^as an ai(?:[,\s]|$)"),
    re.compile(r"^as a language model(?:[,\s]|$)"),
]


# Substring patterns: refusals often appear after a polite preamble
# ("I'm sorry, but I cannot…"). Anchor with \b to avoid mid-word collisions.
_REFUSAL_SUBSTRING_REGEXES = [
    re.compile(r"\b(?:i'?m sorry,? but i (?:cannot|can not|can't|won't))\b"),
    re.compile(r"\b(?:unfortunately,? i (?:cannot|can'?t|won'?t))\b"),
    re.compile(r"\b(?:my apologies,? i (?:cannot|can'?t))\b"),
    re.compile(r"\b(?:i (?:am unable|am not able|won'?t|can'?t|refuse) to (?:help|assist|do))\b"),
    re.compile(r"^sorry,? i (?:cannot|can'?t|won'?t)\b"),
]


def _normalize_for_refusal(text: str) -> str:
    """NFKC-normalize and fold typographic apostrophes to ASCII before
    case-folding. NFKC handles a wide range of compatibility variants;
    the explicit replacements cover apostrophe glyphs that NFKC leaves
    alone (U+2019, U+2018, U+02BC)."""
    nfkc = unicodedata.normalize("NFKC", text)
    folded = nfkc.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    return folded.strip().lower()


def is_refusal(text: str) -> bool:
    """Detect refusal-ish responses.

    Three layers, all case-insensitive after Unicode/apostrophe normalization:
      1. Strong-prefix match against REFUSAL_PHRASES (text starts with phrase).
      2. Bounded prefix regexes ("as an ai," / "as a language model,…")
         that avoid false positives like "as an aid…".
      3. Substring regexes that match polite-preamble refusals anywhere in
         the response ("I'm sorry, but I cannot…", "Unfortunately, I can't…").
    """
    normalized = _normalize_for_refusal(text)
    if any(normalized.startswith(p) for p in REFUSAL_PHRASES):
        return True
    if any(rx.match(normalized) for rx in _REFUSAL_PREFIX_REGEXES):
        return True
    return any(rx.search(normalized) for rx in _REFUSAL_SUBSTRING_REGEXES)


@dataclass(frozen=True)
class AdjustedParaphraseResult:
    """Outcome of one paraphrase request (after retries).

    `retries_used` is the total number of model attempts the loop made
    (success on the first attempt -> 1; full refusal at max_retries=N -> N+1).
    """

    paraphrase: str  # the chosen paraphrase text (or original on full refusal)
    actual_edit_pct: float  # achieved token edit %; 0.0 when refused
    retries_used: int  # total number of model attempts (>=1)
    refused: bool  # True iff every attempt was a refusal


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


_INITIAL_PROMPT = """You are tasked with paraphrasing the text below.

TARGET: Change exactly {target_pct:.1f}% of tokens (±{tolerance:.1f}% tolerance).

HOW WE MEASURE:
We tokenize both texts using a language model tokenizer, then calculate:
  edit_distance(original_tokens, paraphrased_tokens) / len(original_tokens) * 100

CRITICAL REQUIREMENTS:
- The paraphrase MUST be semantically equivalent to the original.
- Preserve ALL technical and factual content EXACTLY.
- You MUST achieve approximately {target_pct:.1f}% token change.

Original text ({orig_token_count} tokens):
{text}

Provide ONLY the paraphrased text, nothing else."""


_RETRY_PROMPT = (
    "That changed {actual_pct:.1f}% of tokens. Target is {target_pct:.1f}% "
    "(±{tolerance:.1f}%). Please produce another paraphrase that gets closer "
    "to the target. Provide ONLY the paraphrased text, nothing else."
)


_REFUSAL_CORRECTION = "That was a refusal. Please provide only the paraphrased text, nothing else."


def generate_adjusted_paraphrase(
    text: str,
    target_edit_pct: float,
    paraphrase_model,
    tokenizer,
    tolerance: float = 0.5,
    max_retries: int = 50,
    seed: int | None = None,
) -> AdjustedParaphraseResult:
    """Iteratively prompt the paraphrase_model to produce a paraphrase whose
    token-edit % is within `tolerance` of `target_edit_pct`.

    On full refusal across all attempts, returns AdjustedParaphraseResult with
    paraphrase = `text` (original) and refused=True. The caller is expected
    to flag baseline_refused=True in baselines_df and emit a WARNING.

    The same `seed` is reused across all retry attempts within a single call:
    paraphrase retries should be deterministic for a given sample.

    See spec §5.3 for the algorithm details.
    """
    orig_ids = tokenizer.encode(text)
    orig_len = len(orig_ids)
    if orig_len == 0:
        raise ValueError("generate_adjusted_paraphrase: text tokenizes to 0 tokens")

    attempts: list[dict] = []
    initial_prompt = _INITIAL_PROMPT.format(
        target_pct=target_edit_pct,
        tolerance=tolerance,
        orig_token_count=orig_len,
        text=text,
    )

    generate_kwargs = {"per_prompt_seeds": [seed]} if seed is not None else {}

    current_prompt = initial_prompt
    for attempt_num in range(max_retries + 1):
        candidate = paraphrase_model.generate([current_prompt], **generate_kwargs)[0].strip()

        if is_refusal(candidate):
            _logger.debug(
                "Paraphrase attempt %d/%d was a refusal", attempt_num + 1, max_retries + 1
            )
            current_prompt = _REFUSAL_CORRECTION
            continue

        actual_pct = token_edit_distance_pct(orig_ids, tokenizer.encode(candidate))
        deviation = abs(actual_pct - target_edit_pct)
        attempts.append({"paraphrase": candidate, "actual_pct": actual_pct, "deviation": deviation})

        if deviation <= tolerance:
            return AdjustedParaphraseResult(
                paraphrase=candidate,
                actual_edit_pct=actual_pct,
                retries_used=attempt_num + 1,
                refused=False,
            )
        current_prompt = _RETRY_PROMPT.format(
            actual_pct=actual_pct,
            target_pct=target_edit_pct,
            tolerance=tolerance,
        )

    if not attempts:
        # Every attempt was a refusal.
        return AdjustedParaphraseResult(
            paraphrase=text,
            actual_edit_pct=0.0,
            retries_used=max_retries + 1,
            refused=True,
        )

    best = select_best_undershoot(attempts, target_pct=target_edit_pct)
    return AdjustedParaphraseResult(
        paraphrase=best["paraphrase"],
        actual_edit_pct=best["actual_pct"],
        retries_used=max_retries + 1,
        refused=False,
    )
