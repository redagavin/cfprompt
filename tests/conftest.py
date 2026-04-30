"""Shared pytest fixtures and reusable stubs.

The Study test stubs (StubTokenizer, StubModel) are defined here so each
integration test file imports them rather than re-declaring local copies.
"""

from __future__ import annotations

import numpy as np


class StubTokenizer:
    """Whitespace tokenizer with a deterministic word-id mapping.

    A class-level vocab keeps ids stable across test runs (Python's built-in
    `hash()` is salted by PYTHONHASHSEED, so the previous local versions
    produced different ids per process).
    """

    _vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for w in text.split():
            if w not in StubTokenizer._vocab:
                StubTokenizer._vocab[w] = len(StubTokenizer._vocab)
            ids.append(StubTokenizer._vocab[w])
        return ids

    @property
    def cache_id(self) -> str:
        return "tok:1"


class StubModel:
    """Model double for Study tests; returns canned probs/generations."""

    def __init__(
        self,
        cache_id: str = "m:1",
        probs_per_call=None,
        gens_per_call=None,
    ):
        self.cache_id = cache_id
        self.tokenizer = StubTokenizer()
        self._probs = list(probs_per_call) if probs_per_call else None
        self._gens = list(gens_per_call) if gens_per_call else None

    def score_classes(self, prompts, classes, per_prompt_seeds=None):
        if self._probs:
            return self._probs.pop(0)
        return np.full((len(prompts), len(classes)), 1.0 / len(classes))

    def generate(self, prompts, per_prompt_seeds=None):
        if self._gens:
            return self._gens.pop(0)
        return [""] * len(prompts)

    def close(self):
        pass
