"""Adjusted paraphrase baseline generator (port of calibrated_paraphrase.py)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .tokenization import token_edit_distance_pct

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

    paraphrase: str  # the chosen paraphrase text (or original on full refusal)
    actual_edit_pct: float  # achieved token edit %; 0.0 when refused
    retries_used: int  # number of retries that produced any paraphrase
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
) -> AdjustedParaphraseResult:
    """Iteratively prompt the paraphrase_model to produce a paraphrase whose
    token-edit % is within `tolerance` of `target_edit_pct`.

    On full refusal across all attempts, returns AdjustedParaphraseResult with
    paraphrase = `text` (original) and refused=True. The caller is expected
    to flag baseline_refused=True in baselines_df and emit a WARNING.

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

    current_prompt = initial_prompt
    for attempt_num in range(max_retries + 1):
        candidate = paraphrase_model.generate([current_prompt])[0].strip()

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
                retries_used=attempt_num,
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
            retries_used=max_retries,
            refused=True,
        )

    best = select_best_undershoot(attempts, target_pct=target_edit_pct)
    return AdjustedParaphraseResult(
        paraphrase=best["paraphrase"],
        actual_edit_pct=best["actual_pct"],
        retries_used=max_retries,
        refused=False,
    )
