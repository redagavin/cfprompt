"""Model ABC and Tokenizer Protocol.

The Model ABC defines the minimum surface that the Study orchestrator,
paraphrase pipeline, and metrics consume. Concrete subclasses (OpenAIModel,
HFModel) live in sibling modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer contract used by paraphrase edit-distance and cache
    keys. Backends should return a thin wrapper around their underlying
    tokenizer rather than the raw library object — `transformers.AutoTokenizer`,
    for example, has no `cache_id` property.
    """

    def encode(self, text: str) -> list[int]:
        """Encode `text` as a list of token IDs."""

    @property
    def cache_id(self) -> str:
        """Stable string identifying tokenizer identity and version, e.g.,
        f'hf:{name_or_path}@{revision_sha}' or f'tiktoken:{encoding_name}'.
        """


class Model(ABC):
    """Abstract base for cfprompt model backends.

    Subclasses MUST implement: generate, score_classes, close, tokenizer,
    cache_id.

    Subclasses are not required to be thread-safe; the Study orchestrator
    invokes their methods serially.
    """

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        per_prompt_seeds: list[int] | None = None,
    ) -> list[str]:
        """Generate text for each prompt. Return one string per prompt.

        Maximum tokens is configured at construction time
        (HFModel.max_new_tokens, OpenAIModel.max_completion_tokens). Both
        subclasses must include their max-token kwarg in cache_id so
        changing it invalidates the inference cache.
        """

    @abstractmethod
    def score_classes(
        self,
        prompts: list[str],
        classes: list[str],
        per_prompt_seeds: list[int] | None = None,
    ) -> np.ndarray:
        """Return a (len(prompts), len(classes)) array of class probabilities.

        Rows sum to 1. Columns are aligned to the user-supplied `classes`
        list order (no internal sorting).
        """

    @abstractmethod
    def close(self) -> None:
        """Release model resources (GPU memory, connections). Idempotent."""

    @property
    @abstractmethod
    def tokenizer(self) -> Tokenizer:
        """Tokenizer wrapper for this backend."""

    @property
    @abstractmethod
    def cache_id(self) -> str:
        """Stable string identifying everything that affects model outputs.
        See OpenAIModel.cache_id and HFModel.cache_id docstrings for the
        exact formats."""

    def __enter__(self) -> Model:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
