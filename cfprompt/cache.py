"""Content-addressed per-sample cache for cfprompt.

This module owns:
- derive_seed: deterministic seed derivation across processes/Python versions
- (later) cache key composition, atomic file IO, hash-prefix sharding
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


_AcceptedSeedPart = (int, str, tuple)


def _check_seed_part(p: Any) -> None:
    if isinstance(p, bool):
        # bool is a subclass of int, but we want explicit int only
        raise TypeError(
            f"derive_seed parts must be int, str, or tuple thereof; got bool ({p!r})"
        )
    if isinstance(p, int):
        return
    if isinstance(p, str):
        return
    if isinstance(p, tuple):
        for sub in p:
            _check_seed_part(sub)
        return
    raise TypeError(
        f"derive_seed parts must be int, str, or tuple thereof; got "
        f"{type(p).__name__} ({p!r})"
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
        raise TypeError(
            f"study_seed must be int; got {type(study_seed).__name__}"
        )
    for p in parts:
        _check_seed_part(p)
    blob = json.dumps([study_seed, *parts], sort_keys=True).encode("utf-8")
    raw = int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")
    return raw & ((1 << 63) - 1)
