# ABOUTME: OpenAIModel and TiktokenWrapper - Chat Completions-backed Model implementation.
# ABOUTME: TiktokenWrapper wraps tiktoken encodings to expose the Tokenizer Protocol.
"""OpenAIModel: Chat Completions-backed Model implementation."""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from openai import OpenAI

from .base import Model, Tokenizer

_logger = logging.getLogger("cfprompt")


@dataclass
class TiktokenWrapper:
    """Tokenizer Protocol implementation backed by tiktoken."""

    encoding_name: str
    _enc: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._enc = tiktoken.get_encoding(self.encoding_name)

    def encode(self, text: str) -> list[int]:
        return list(self._enc.encode(text))

    @property
    def cache_id(self) -> str:
        return f"tiktoken:{self.encoding_name}"


class OpenAIModel(Model):
    """OpenAI Chat Completions-backed Model.

    See spec section 5.2 for the full contract:
      - Static preflight (single-token + <= 20 classes) fires at Study.__init__,
        not here.
      - Dynamic missing-class drops the affected sample; never epsilon-fallbacks.
      - cache_id includes name+temp+top_p+max_completion_tokens+class_prefix+
        system_fingerprint.
      - Warm-up at __init__ resolves system_fingerprint via a 1-token call.
        Offline / missing-fp -> per-instance UUID to prevent cross-process
        cache contamination.
      - HTTP retry budget: 10 with exponential backoff capped at 60s, on
        429 / 5xx / timeout / connection errors. 4xx other than 429 raises
        immediately (after sanitizing).
    """

    def __init__(
        self,
        name: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 512,
        max_concurrent: int = 8,
        timeout: float = 60.0,
        max_retries: int = 10,
        backoff_max_seconds: float = 60.0,
        class_prefix: str = " ",
    ) -> None:
        self.name = name
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_max_seconds = backoff_max_seconds
        # OpenAI tokenizers are BPE: most chat completions emit class tokens
        # with a leading space (e.g., " A" not "A"). class_prefix is prepended
        # to user-supplied class strings before the static-preflight tokenize
        # check AND before the dynamic top_logprobs lookup, mirroring HFModel.
        self.class_prefix = class_prefix
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # tiktoken encoding is resolved lazily; AutoTokenizer-style mapping
        # from model name to encoding is in `_resolve_encoding`.
        self._tokenizer_wrapper = TiktokenWrapper(self._resolve_encoding(name))

        # Warm-up resolves system_fingerprint (or sets a per-instance UUID
        # on offline / missing-fp).
        self._system_fingerprint: str | None = None
        try:
            self._warmup()
        except Exception as e:
            _logger.warning(
                "OpenAIModel(%r) warm-up failed (%s); cache_id will use a "
                "per-instance UUID and gain no offline cache reuse.",
                name,
                e,
            )
            self._system_fingerprint = None

        if self._system_fingerprint in (None, ""):
            self._system_fingerprint = f"unknown-{uuid.uuid4().hex}"

    @staticmethod
    def _resolve_encoding(name: str) -> str:
        """Map a model name to a tiktoken encoding name."""
        try:
            enc = tiktoken.encoding_for_model(name)
            return enc.name
        except KeyError:
            # Fallback: o200k_base covers gpt-4o family and newer.
            return "o200k_base"

    def _warmup(self) -> None:
        """Issue a 1-token completion to resolve system_fingerprint.

        Subclasses/tests can monkeypatch this to inject a fingerprint or
        simulate offline failure.
        """
        client = OpenAI(api_key=self._api_key, timeout=self.timeout)
        resp = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": "ok"}],
            max_completion_tokens=1,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self._system_fingerprint = getattr(resp, "system_fingerprint", None)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer_wrapper

    @property
    def cache_id(self) -> str:
        return (
            f"openai:{self.name}"
            f"|temp={self.temperature}"
            f"|top_p={self.top_p}"
            f"|max_tok={self.max_completion_tokens}"
            f"|class_prefix={self.class_prefix!r}"
            f"|fp={self._system_fingerprint}"
        )

    def close(self) -> None:
        # OpenAI SDK clients are httpx-backed; allow GC to handle them.
        pass

    def generate(self, prompts, per_prompt_seeds=None):
        raise NotImplementedError("Implemented in Task 4.4")

    def score_classes(self, prompts, classes, per_prompt_seeds=None):
        raise NotImplementedError("Implemented in Task 4.3")
