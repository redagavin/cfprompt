"""Content-addressed per-sample cache for cfprompt.

This module owns:
- derive_seed: deterministic seed derivation across processes/Python versions
- (later) cache key composition, atomic file IO, hash-prefix sharding
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pickle
import string
import struct
import tempfile
import time
from pathlib import Path
from typing import Any

from .exceptions import ConfigError

logger = logging.getLogger(__name__)


def _check_seed_part(p: Any) -> None:
    if isinstance(p, bool):
        # bool is a subclass of int, but we want explicit int only
        raise TypeError(f"derive_seed parts must be int, str, or tuple thereof; got bool ({p!r})")
    if isinstance(p, int):
        return
    if isinstance(p, str):
        return
    if isinstance(p, tuple):
        for sub in p:
            _check_seed_part(sub)
        return
    raise TypeError(
        f"derive_seed parts must be int, str, or tuple thereof; got {type(p).__name__} ({p!r})"
    )


def derive_seed(study_seed: int, *parts: Any) -> int:
    """Deterministic seed derivation.

    Stable across Python versions and processes (unlike built-in hash(),
    which is randomized per process for str/bytes).

    `parts` MUST be primitive JSON-serializable types — int, str, or tuples
    thereof. bytes, np.int64, Path, dict, list, and custom objects raise
    TypeError to prevent silent stringification ambiguity.

    Returns a non-negative signed-int63 value (0 <= x < 2^63) so the result
    can be passed to APIs (e.g., OpenAI seed) that reject values exceeding
    the signed int64 max.
    """
    if not isinstance(study_seed, int) or isinstance(study_seed, bool):
        raise TypeError(f"study_seed must be int; got {type(study_seed).__name__}")
    for p in parts:
        _check_seed_part(p)
    blob = json.dumps([study_seed, *parts], sort_keys=True).encode("utf-8")
    raw = int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")
    return raw & ((1 << 63) - 1)


def safe_format(template: str, mapping: dict) -> str:
    """Substitute `{name}` placeholders in `template` from `mapping`.

    Strict policy:
    - Only bare `{name}` and `{{` / `}}` (literal brace escapes) are allowed.
    - Rejects: attribute access ({x.foo}), indexing ({x[0]}),
      conversion ({x!r}), format spec ({x:>10}), nested specs ({x:{y}}),
      positional/numeric placeholders ({}, {0}).
    - Literal `{` and `}` inside mapping VALUES are NOT recursively interpreted.
    - Missing placeholder raises ConfigError naming the offending key and
      the available columns.

    Implementation: walks `string.Formatter.parse(template)` to validate each
    field tuple, then performs the substitution with simple string replacement.
    """
    formatter = string.Formatter()
    try:
        parsed = list(formatter.parse(template))
    except ValueError as e:
        raise ConfigError(f"prompt_template is not valid format string: {e}") from e

    keys_used: list[str] = []
    for _literal_text, field_name, format_spec, conversion in parsed:
        if field_name is None:
            # pure literal segment (or after the last placeholder)
            continue
        if field_name == "":
            raise ConfigError(
                "prompt_template uses a positional placeholder ('{}'); "
                "only named placeholders ({name}) are allowed."
            )
        if field_name.isdigit():
            raise ConfigError(
                f"prompt_template uses a numeric/positional placeholder "
                f"('{{{field_name}}}'); only named placeholders ({{name}}) "
                f"are allowed."
            )
        if "." in field_name:
            raise ConfigError(
                f"prompt_template attempts attribute access in '{{{field_name}}}'; "
                f"only bare {{name}} placeholders are allowed."
            )
        if "[" in field_name:
            raise ConfigError(
                f"prompt_template attempts indexing in '{{{field_name}}}'; "
                f"only bare {{name}} placeholders are allowed."
            )
        if conversion is not None:
            raise ConfigError(
                f"prompt_template uses a conversion ('!{conversion}') in "
                f"'{{{field_name}!{conversion}}}'; conversions are not allowed."
            )
        if format_spec != "":
            raise ConfigError(
                f"prompt_template uses a format spec (':{format_spec}') in "
                f"'{{{field_name}:{format_spec}}}'; format specs (and nested "
                f"format specs) are not allowed."
            )
        keys_used.append(field_name)

    missing = [k for k in keys_used if k not in mapping]
    if missing:
        available = sorted(mapping.keys())
        raise ConfigError(
            f"prompt_template references {missing[0]!r} but data has no such "
            f"column. Missing keys: {missing}. Available: {available}."
        )

    # Walk the parsed pieces and reassemble. Use string concat (NOT
    # `template.format(**mapping)`), because `.format` would re-interpret
    # literal braces inside mapping values.
    out_parts: list[str] = []
    for literal_text, field_name, _format_spec, _conversion in parsed:
        out_parts.append(literal_text)
        if field_name is not None:
            value = mapping[field_name]
            out_parts.append(str(value))
    return "".join(out_parts)


def paraphrase_cache_key(
    *,
    stage_version: str,
    cfprompt_version: str,
    original: str,
    target_perturbed: str,
    target_edit_pct: float,
    paraphrase_model_cache_id: str,
    tokenizer_cache_id: str,
    tolerance: float,
    max_retries: int,
    seed: int,
) -> str:
    """SHA-256 hex digest of canonical JSON of all paraphrase inputs.

    Note on ``target_edit_pct``: rounded to 4 decimals when hashed. The
    paraphrase loop's tolerance is well above 1e-4 (default 0.5pp = 5e-3),
    so 4 decimals provides ample headroom while preventing spurious cache
    misses from tiny float-representation differences in the same target.
    """
    blob = json.dumps(
        {
            "stage_version": stage_version,
            "cfprompt_version": cfprompt_version,
            "original": original,
            "target_perturbed": target_perturbed,
            "target_edit_pct": round(float(target_edit_pct), 4),
            "paraphrase_model_cache_id": paraphrase_model_cache_id,
            "tokenizer_cache_id": tokenizer_cache_id,
            "tolerance": float(tolerance),
            "max_retries": int(max_retries),
            "seed": int(seed),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def inference_cache_key(
    *,
    stage_version: str,
    prompt: str,
    target_model_cache_id: str,
    mode: str,
    classes: list[str] | None,
    seed: int,
) -> str:
    """SHA-256 hex digest of canonical JSON of all inference inputs.

    `classes` is preserved in user-supplied order (not sorted).
    """
    blob = json.dumps(
        {
            "stage_version": stage_version,
            "prompt": prompt,
            "target_model_cache_id": target_model_cache_id,
            "mode": mode,
            "classes": list(classes) if classes is not None else None,
            "seed": int(seed),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class DiskCache:
    """Pickle-backed, hash-prefix-sharded, atomic-write cache.

    Trust caveat: the cache uses pickle. Loading a cache directory written
    by another (untrusted) process is equivalent to running their code.
    Treat cache_dir as user-private. See spec §6.6.
    """

    MISS = object()

    def __init__(self, cache_dir: str | Path, namespace: str) -> None:
        self.root = Path(cache_dir) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Cache keys are always 64-char lowercase hex sha256 digests.
        if len(key) != 64:
            raise ValueError(
                f"cache key must be a 64-char sha256 hex digest; got len={len(key)}: {key!r}"
            )
        return self.root / key[:2] / key[2:4] / f"{key}.pkl"

    def get(self, key: str, default=None):
        """Read the cached value at `key`. Returns `default` (None by default,
        matching dict-like convention) ONLY when the file does not exist. A
        successfully-cached `None` value is returned as `None` by default —
        which is indistinguishable from a miss. Call sites that need to
        disambiguate (e.g., `_run_inference` storing `None` for
        OpenAI dynamic missing-class) pass `default=DiskCache.MISS`.

        On a corrupt cache file (EOFError / UnpicklingError / struct.error),
        retries once after a short sleep; if still corrupt, logs a WARNING
        and returns `default` (treating the file as a miss)."""
        path = self._path(key)
        for _attempt in range(2):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                return default
            except (EOFError, pickle.UnpicklingError, struct.error):
                time.sleep(0.05)
        logger.warning("cache file at %s appears corrupt; treating as miss", path)
        return default

    def has(self, key: str) -> bool:
        """Cheap presence check (no deserialization)."""
        return self._path(key).exists()

    def set(self, key: str, value) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=f"{key}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            # Spec §6: write tmp, fsync, os.replace(tmp, final). The fsync
            # guarantees the file's bytes hit the disk platter before the
            # atomic rename, so a crash between fdopen and os.replace cannot
            # leave a half-written tmp file masquerading as the final value.
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(value, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise
