# ABOUTME: HFModel and HFTokenizer — transformers-backed Model implementation.
# ABOUTME: HFTokenizer wraps AutoTokenizer to expose cache_id pinned to a revision SHA.
"""HFModel: transformers-backed Model implementation."""

from __future__ import annotations

import contextlib
import gc
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

from cfprompt.exceptions import ClassificationModeError

from .base import Model, Tokenizer

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
    revision_label: str  # original `revision` kwarg ("main" / "v1" / SHA)
    resolved_sha: str | None  # SHA when available, else None (fallback to literal label)
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


class HFModel(Model):
    """transformers-backed Model.

    See spec §5.2 for the full contract:
      - score_classes uses `class_prefix + class` first-token id at the last
        prompt position; raises ClassificationModeError on first-token
        collisions.
      - generate wraps stochastic decoding in torch.random.fork_rng for
        per-prompt seeds; falls back to batch=1 when seeds are heterogeneous.
      - cache_id includes name+revision_sha+dtype+class_prefix.
      - close() drops weights and calls torch.cuda.empty_cache().
    """

    def __init__(
        self,
        name_or_path: str,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        batch_size: int = 8,
        max_new_tokens: int = 512,
        device_map: str | dict | None = None,
        revision: str = "main",
        class_prefix: str = " ",
    ) -> None:
        from transformers import AutoModelForCausalLM

        self.name_or_path = name_or_path
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.revision = revision
        self.class_prefix = class_prefix

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._tokenizer_wrapper = HFTokenizer.from_name_and_revision(
            name_or_path, revision=revision
        )
        kwargs = {"torch_dtype": dtype, "revision": revision}
        if device_map is not None:
            kwargs["device_map"] = device_map
        self._model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        if device_map is None:
            self._model = self._model.to(self.device)
        self._model.eval()

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer_wrapper

    @property
    def cache_id(self) -> str:
        return (
            f"{self._tokenizer_wrapper.cache_id}"
            f"|dtype={self.dtype}"
            f"|class_prefix={self.class_prefix!r}"
        )

    def close(self) -> None:
        if self._model is not None:
            with contextlib.suppress(Exception):
                self._model.cpu()
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _resolve_sampling_device(self) -> torch.device:
        """Walk a fallback chain to find where sampling happens.

        Order: lm_head.weight.device -> output_projection.weight.device ->
        first model parameter's device -> cpu fallback (with DEBUG log).
        Robust under device_map sharding, PEFT wrappers, quantized models,
        and the (rare) zero-parameter case.
        """
        m = self._model
        for attr in ("lm_head", "output_projection"):
            mod = getattr(m, attr, None)
            if mod is not None:
                weight = getattr(mod, "weight", None)
                if weight is not None:
                    return weight.device
        first = next(m.parameters(), None)
        if first is not None:
            return first.device
        _logger.debug(
            "HFModel: model has no parameters; falling back to cpu for sampling-device resolution"
        )
        return torch.device("cpu")

    def generate(
        self,
        prompts,
        per_prompt_seeds=None,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        underlying = self._tokenizer_wrapper._tokenizer
        n = len(prompts)
        if per_prompt_seeds is not None and len(per_prompt_seeds) != n:
            raise ValueError(
                f"per_prompt_seeds length {len(per_prompt_seeds)} does not "
                f"match number of prompts {n}"
            )

        sampling_device = self._resolve_sampling_device()

        def _decode_one_batch(chunk, seeds_chunk=None):
            """Run a forward pass on one chunk and decode the new tokens.
            Padding is LEFT-side (configured in HFTokenizer.__post_init__),
            so each row's prompt ends at column -1 and `prompt_len` is
            constant across rows in the batch (= input_ids.shape[1]).
            """
            enc = underlying(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(sampling_device)
            attention_mask = enc["attention_mask"].to(sampling_device)
            prompt_len = input_ids.shape[1]
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                pad_token_id=underlying.pad_token_id or underlying.eos_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
            with torch.no_grad():
                out_ids = self._model.generate(**gen_kwargs)
            new_tokens = out_ids[:, prompt_len:]
            return [underlying.decode(row, skip_special_tokens=True) for row in new_tokens]

        # Stochastic decoding with heterogeneous per-prompt seeds -> batch=1
        # with fork_rng so seed setting doesn't bleed into the rest of the
        # process. Greedy or no-seeds -> batched.
        results: list[str] = []
        if do_sample and per_prompt_seeds is not None:
            for i, prompt in enumerate(prompts):
                seed_i = int(per_prompt_seeds[i])
                if sampling_device.type == "cuda":
                    cuda_idx = (
                        sampling_device.index
                        if sampling_device.index is not None
                        else torch.cuda.current_device()
                    )
                    cuda_devices = [cuda_idx]
                else:
                    cuda_idx = None
                    cuda_devices = []
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(seed_i)
                    if cuda_devices:
                        # `torch.cuda.manual_seed` seeds the CURRENT device,
                        # not necessarily `cuda_idx`. Under multi-GPU
                        # (sampling on cuda:1 while current_device==0), we
                        # must explicitly switch the current device so the
                        # seed lands on the device whose RNG we forked.
                        # NOT `manual_seed_all` — that would seed all CUDA
                        # devices, mutating state we did NOT fork (= leak).
                        with torch.cuda.device(cuda_idx):
                            torch.cuda.manual_seed(seed_i)
                    results.extend(_decode_one_batch([prompt]))
        else:
            # Greedy decoding (deterministic) or no seeds: batched.
            if do_sample is False and per_prompt_seeds is not None:
                _logger.debug(
                    "HFModel.generate: per_prompt_seeds provided but "
                    "do_sample=False; seeds are recorded in the inference "
                    "cache key but do not affect outputs (forward pass is "
                    "deterministic)."
                )
            bs = self.batch_size
            for start in range(0, n, bs):
                chunk = prompts[start : start + bs]
                results.extend(_decode_one_batch(chunk))
        return results

    def validate_classes(self, classes: list[str]) -> list[int]:
        """Resolve and validate first-token ids for `classes` under the
        configured class_prefix. Raises ClassificationModeError on empty
        tokenization or first-token collisions. Returns the resolved
        first-token ids in input order."""
        underlying = self._tokenizer_wrapper._tokenizer
        first_token_ids: list[int] = []
        for cls in classes:
            ids = underlying.encode(self.class_prefix + cls, add_special_tokens=False)
            if not ids:
                raise ClassificationModeError(
                    f"Class {cls!r} tokenizes to an empty sequence under "
                    f"{self._tokenizer_wrapper.cache_id} (with class_prefix="
                    f"{self.class_prefix!r}); cannot score."
                )
            first_token_ids.append(ids[0])

        seen: dict[int, str] = {}
        for cls, tid in zip(classes, first_token_ids, strict=False):
            if tid in seen:
                hint = (
                    " Try class_prefix='' (SentencePiece tokenizers like "
                    "Llama 1/2 collapse a leading space to the single '▁' "
                    "token, which causes this collision)."
                    if self.class_prefix == " "
                    else ""
                )
                raise ClassificationModeError(
                    f"Classes {seen[tid]!r} and {cls!r} both tokenize to "
                    f"first token id {tid} under "
                    f"{self._tokenizer_wrapper.cache_id} with class_prefix="
                    f"{self.class_prefix!r}; first-token scoring would "
                    f"conflate them.{hint} Otherwise choose distinct class "
                    f"names or use free-form mode with a custom extract_label."
                )
            seen[tid] = cls
        return first_token_ids

    def score_classes(self, prompts, classes, per_prompt_seeds=None):
        underlying = self._tokenizer_wrapper._tokenizer
        first_token_ids = self.validate_classes(classes)

        # Forward pass, batched.
        # HFTokenizer is configured with padding_side="left", so the final
        # real token is always at index -1. This avoids per-row indexing
        # via attention_mask.sum(dim=1)-1 which is brittle across HF versions.
        n = len(prompts)
        k = len(classes)
        out = np.empty((n, k), dtype=np.float64)
        bs = self.batch_size
        device = self._resolve_sampling_device()
        for start in range(0, n, bs):
            chunk = prompts[start : start + bs]
            enc = underlying(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
            # Last position is index -1 because padding is left-side.
            last_logits = logits[:, -1, :]  # (batch, vocab)
            class_logits = last_logits[:, first_token_ids]  # (batch, k)
            class_probs = torch.softmax(class_logits.float(), dim=-1).cpu().numpy()
            out[start : start + class_probs.shape[0]] = class_probs

        return out
