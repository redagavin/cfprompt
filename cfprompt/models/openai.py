# ABOUTME: OpenAIModel and TiktokenWrapper - Chat Completions-backed Model implementation.
# ABOUTME: TiktokenWrapper wraps tiktoken encodings to expose the Tokenizer Protocol.
"""OpenAIModel: Chat Completions-backed Model implementation."""

from __future__ import annotations

import contextlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tiktoken
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .base import Model, Tokenizer

_logger = logging.getLogger("cfprompt")


def _safe_get_error_code(e: Exception) -> str | None:
    """Extract an OpenAI error code from a raised exception.

    APIStatusError.body may be a dict, a pydantic-style object with a `code`
    attribute, None, or any other type. Treat anything we can't safely read
    as missing rather than crashing the error-logging path.
    """
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        return body.get("code")
    return getattr(body, "code", None)


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
        timeout: float = 60.0,
        max_retries: int = 10,
        backoff_max_seconds: float = 60.0,
        class_prefix: str = " ",
    ) -> None:
        self.name = name
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
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

        # Track every system_fingerprint we observe across all requests on
        # this instance. A new value (different from the warmup-resolved
        # one) indicates the backend rolled to a new model snapshot mid-run,
        # which can change outputs and invalidate cache reuse. Warn once
        # per new fingerprint.
        self._observed_fingerprints: set[str] = set()
        self._closed: bool = False

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

    def _record_fingerprint(self, fp: str | None) -> None:
        """Compare a response's system_fingerprint against the warmup-resolved
        one and warn (once per new fingerprint) on drift."""
        if not fp:
            return
        if fp == self._system_fingerprint:
            return
        if fp in self._observed_fingerprints:
            return
        self._observed_fingerprints.add(fp)
        _logger.warning(
            "system_fingerprint drift: observed %r, init resolved %r",
            fp,
            self._system_fingerprint,
        )

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

        Sampling parameters (temperature, top_p) are intentionally omitted:
        reasoning models (o1, o3, gpt-5) reject them with a 400, and warmup
        only needs the fingerprint, not deterministic output.

        Subclasses/tests can monkeypatch this to inject a fingerprint or
        simulate offline failure.
        """
        client = OpenAI(api_key=self._api_key, timeout=self.timeout)
        resp = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": "ok"}],
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
        """Close the underlying OpenAI client (httpx connection pool) and
        block subsequent requests. Idempotent."""
        if self._closed:
            return
        self._closed = True
        client = getattr(self, "_client", None)
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()
            self._client = None

    def _get_client(self) -> "OpenAI":
        """Return a process-lifetime OpenAI client. Lazy so warmup paths and
        tests that don't actually call create() don't open a connection
        pool."""
        if self._closed:
            raise RuntimeError("model closed")
        client = getattr(self, "_client", None)
        if client is None:
            client = OpenAI(api_key=self._api_key, timeout=self.timeout)
            self._client = client
        return client

    def _request_one_generate(self, prompt: str, seed: int | None) -> dict:
        """Issue one free-form generation request. Same retry policy as
        _request_one but uses max_completion_tokens for output length and
        does NOT request logprobs."""
        client = self._get_client()
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": self.max_completion_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
                if seed is not None:
                    kwargs["seed"] = seed
                resp = client.chat.completions.create(**kwargs)
                self._record_fingerprint(getattr(resp, "system_fingerprint", None))
                return resp.model_dump()
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt == self.max_retries:
                    _logger.error(
                        "OpenAIModel: exhausted %d retries on %s",
                        self.max_retries,
                        type(e).__name__,
                    )
                    raise
                wait = min(delay, self.backoff_max_seconds)
                _logger.warning(
                    "OpenAIModel: %s (attempt %d/%d); retrying in %.1fs",
                    type(e).__name__,
                    attempt + 1,
                    self.max_retries + 1,
                    wait,
                )
                time.sleep(wait)
                delay = min(delay * 2, self.backoff_max_seconds)
            except APIStatusError as e:
                if 500 <= e.status_code < 600 and attempt < self.max_retries:
                    wait = min(delay, self.backoff_max_seconds)
                    _logger.warning(
                        "OpenAIModel: %d (attempt %d/%d); retrying in %.1fs",
                        e.status_code,
                        attempt + 1,
                        self.max_retries + 1,
                        wait,
                    )
                    time.sleep(wait)
                    delay = min(delay * 2, self.backoff_max_seconds)
                    continue
                err_code = _safe_get_error_code(e)
                msg = str(getattr(e, "message", "")) or str(e)
                _logger.error(
                    "OpenAIModel: %d %s %s",
                    e.status_code,
                    err_code or "",
                    msg[:200],
                )
                raise

    def generate(self, prompts, per_prompt_seeds=None):
        out: list[str] = []
        for i, prompt in enumerate(prompts):
            seed = per_prompt_seeds[i] if per_prompt_seeds is not None else None
            resp = self._request_one_generate(prompt, seed)
            try:
                text = resp["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = ""
            out.append(text or "")
        return out

    def _request_one(
        self,
        prompt: str,
        classes: list[str],
        seed: int | None,
    ) -> dict:
        """Issue one scoring request and return the parsed response dict.

        Subclasses/tests can monkeypatch this to inject canned responses.
        Implements HTTP retry policy (429 / 5xx / timeout / network errors)
        with exponential backoff capped at backoff_max_seconds.
        """
        client = self._get_client()
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 1,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "logprobs": True,
                    # Always request the maximum (20) so low-prior classes are
                    # still scored when the model emits unrelated higher-prob
                    # tokens. Capping at len(classes) silently drops valid
                    # classes that fall below the kth most-likely token.
                    "top_logprobs": 20,
                }
                if seed is not None:
                    kwargs["seed"] = seed
                resp = client.chat.completions.create(**kwargs)
                self._record_fingerprint(getattr(resp, "system_fingerprint", None))
                return resp.model_dump()
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt == self.max_retries:
                    _logger.error(
                        "OpenAIModel: exhausted %d retries on %s",
                        self.max_retries,
                        type(e).__name__,
                    )
                    raise
                wait = min(delay, self.backoff_max_seconds)
                _logger.warning(
                    "OpenAIModel: %s (attempt %d/%d); retrying in %.1fs",
                    type(e).__name__,
                    attempt + 1,
                    self.max_retries + 1,
                    wait,
                )
                time.sleep(wait)
                delay = min(delay * 2, self.backoff_max_seconds)
            except APIStatusError as e:
                if 500 <= e.status_code < 600 and attempt < self.max_retries:
                    wait = min(delay, self.backoff_max_seconds)
                    _logger.warning(
                        "OpenAIModel: %d (attempt %d/%d); retrying in %.1fs",
                        e.status_code,
                        attempt + 1,
                        self.max_retries + 1,
                        wait,
                    )
                    time.sleep(wait)
                    delay = min(delay * 2, self.backoff_max_seconds)
                    continue
                # 4xx other than rate-limit: sanitize and re-raise.
                # Note: never log request headers or full body — they may
                # contain Authorization or PHI. We log only status, error
                # code (if available), and the first 200 chars of the
                # message body, after coercing to str.
                err_code = _safe_get_error_code(e)
                msg = str(getattr(e, "message", "")) or str(e)
                _logger.error(
                    "OpenAIModel: %d %s %s",
                    e.status_code,
                    err_code or "",
                    msg[:200],
                )
                raise

    def _extract_class_logprobs(self, response: dict, classes: list[str]) -> np.ndarray:
        """Extract per-class logprobs from a chat-completions response.

        OpenAI's `top_logprobs` returns tokens with the leading-space prefix
        (e.g., `" A"` not `"A"`). We look up `class_prefix + cls` so the
        prefix matches what the API actually returns.

        Returns a (k,) array of logprobs, with NaN for any class missing from
        top_logprobs (caller treats NaN row as missing-class drop).
        """
        try:
            content_logprobs = response["choices"][0]["logprobs"]["content"]
            top = content_logprobs[0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            return np.full(len(classes), np.nan)

        # Build a token->logprob map. OpenAI may include duplicates if the same
        # token appears twice; setdefault keeps the first occurrence (highest
        # logprob, since top_logprobs is sorted descending).
        token_to_logprob: dict[str, float] = {}
        for entry in top:
            token_to_logprob.setdefault(entry["token"], entry["logprob"])

        out = np.empty(len(classes), dtype=np.float64)
        for i, cls in enumerate(classes):
            key = self.class_prefix + cls
            # Strict: only the prefixed form is accepted. Falling back to the
            # bare form would defeat the static-preflight prefix check (a
            # collision that's invisible under one form may be valid under the
            # other), so a missing prefixed token is signalled as NaN and the
            # Study orchestrator drops the row.
            if key in token_to_logprob:
                out[i] = token_to_logprob[key]
            else:
                out[i] = np.nan
        return out

    def score_classes(self, prompts, classes, per_prompt_seeds=None):
        n = len(prompts)
        k = len(classes)
        out = np.empty((n, k), dtype=np.float64)
        for i, prompt in enumerate(prompts):
            seed = per_prompt_seeds[i] if per_prompt_seeds is not None else None
            resp = self._request_one(prompt, classes, seed)
            logprobs = self._extract_class_logprobs(resp, classes)
            if np.isnan(logprobs).any():
                # Dynamic missing-class: signal the entire row as NaN so the
                # Study orchestrator drops the sample (drop_counts.openai_missing_class).
                out[i] = np.nan
            else:
                # Renormalize via softmax over the class subset.
                m = logprobs.max()
                ex = np.exp(logprobs - m)
                out[i] = ex / ex.sum()
        return out
