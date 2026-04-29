# ABOUTME: OpenAIModel and TiktokenWrapper - Chat Completions-backed Model implementation.
# ABOUTME: TiktokenWrapper wraps tiktoken encodings to expose the Tokenizer Protocol.
"""OpenAIModel: Chat Completions-backed Model implementation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import tiktoken

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
