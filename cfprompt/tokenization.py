"""Token-level edit distance + Tokenizer Protocol.

The Tokenizer Protocol itself is defined in `cfprompt.models.base`; this
module owns the pure-numeric edit-distance computation that consumes any
Protocol-conforming tokenizer's `encode` output.
"""
from __future__ import annotations

from collections.abc import Sequence


def token_edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Levenshtein edit distance over two integer token-id sequences.

    Counts insertions, deletions, and substitutions. Each edit costs 1.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # Two-row DP: O(min(n, m)) memory.
    if m < n:
        a, b = b, a
        n, m = m, n

    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for j in range(1, m + 1):
        curr[0] = j
        bj = b[j - 1]
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == bj else 1
            curr[i] = min(
                prev[i] + 1,        # deletion
                curr[i - 1] + 1,    # insertion
                prev[i - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[n]


def token_edit_distance_pct(
    original_tokens: Sequence[int],
    modified_tokens: Sequence[int],
) -> float:
    """Edit distance as a percentage of original length.

        pct = edit_distance(original, modified) / len(original) * 100

    Raises ValueError if `original_tokens` is empty.
    """
    if len(original_tokens) == 0:
        raise ValueError(
            "token_edit_distance_pct: original_tokens is empty; cannot "
            "compute a percentage with zero denominator."
        )
    d = token_edit_distance(original_tokens, modified_tokens)
    return d / len(original_tokens) * 100.0
