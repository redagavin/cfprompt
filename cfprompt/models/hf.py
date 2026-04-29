# ABOUTME: HFModel and HFTokenizer — transformers-backed Model implementation.
# ABOUTME: HFTokenizer wraps AutoTokenizer to expose cache_id pinned to a revision SHA.
"""HFModel: transformers-backed Model implementation."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

if TYPE_CHECKING:
    from transformers import AutoTokenizer as _AutoTokenizer

_logger = logging.getLogger("cfprompt")


def _resolve_revision_sha(name_or_path: str, revision: str) -> str:
    """Resolve `revision` to a 40-char commit SHA via huggingface_hub.

    Tries:
      1. HfApi().repo_info(...).sha    (network)
      2. snapshot_download(local_files_only=True) + read commit ref (offline)

    Raises OSError if both fail. Caller (HFTokenizer) catches and falls back
    to the literal revision string.
    """
    try:
        info = HfApi().repo_info(repo_id=name_or_path, revision=revision)
        if info.sha:
            return info.sha
    except (HfHubHTTPError, OfflineModeIsEnabled, OSError) as e:
        _logger.debug("HfApi.repo_info failed: %s; trying local snapshot", e)
    # Fallback: find a previously-downloaded snapshot for this revision.
    try:
        path = snapshot_download(
            repo_id=name_or_path,
            revision=revision,
            local_files_only=True,
        )
        # snapshot_download returns the local path; the directory name typically
        # encodes the SHA when it was downloaded by SHA. If revision was a tag
        # or branch ("main"), this may not give us a SHA — return path-derived
        # marker.
        # Best-effort: extract from path basename if it looks like a SHA.
        base = os.path.basename(path.rstrip(os.sep))
        if len(base) == 40 and all(c in "0123456789abcdef" for c in base):
            return base
        # Else: re-raise so caller falls back to literal `revision`.
    except (HfHubHTTPError, OfflineModeIsEnabled, OSError, FileNotFoundError) as e:
        _logger.debug("snapshot_download local-only failed: %s", e)
    raise OSError(
        f"could not resolve revision SHA for {name_or_path}@{revision} "
        f"(tried online and local cache)"
    )


@dataclass
class HFTokenizer:
    """Thin wrapper exposing the Tokenizer Protocol over a transformers
    AutoTokenizer (which lacks a `cache_id` of its own)."""

    name_or_path: str
    revision_label: str           # original `revision` kwarg ("main" / "v1" / SHA)
    resolved_sha: str | None      # SHA when available, else None (fallback to literal label)
    _tokenizer: _AutoTokenizer

    @classmethod
    def from_name_and_revision(
        cls,
        name_or_path: str,
        revision: str = "main",
    ) -> HFTokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
        try:
            sha = _resolve_revision_sha(name_or_path, revision)
        except OSError as e:
            _logger.warning(
                "Could not resolve HF revision SHA for %s@%s (%s); cache_id "
                "will not be pinned to a specific commit and may share cache "
                "with future revisions of the same branch.",
                name_or_path,
                revision,
                e,
            )
            sha = None
        return cls(
            name_or_path=name_or_path,
            revision_label=revision,
            resolved_sha=sha,
            _tokenizer=tok,
        )

    def __post_init__(self) -> None:
        # Many causal-LM tokenizers (Llama, GPT-2, etc.) ship without a
        # pad_token. Set it to eos so `padding=True` works downstream.
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Causal-LM scoring/generation requires LEFT padding so that the
        # last real token is at index -1 in every batch row.
        self._tokenizer.padding_side = "left"

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    @property
    def cache_id(self) -> str:
        suffix = self.resolved_sha if self.resolved_sha is not None else self.revision_label
        return f"hf:{self.name_or_path}@{suffix}"
