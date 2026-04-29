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


import string

from .exceptions import ConfigError


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
    for literal_text, field_name, format_spec, conversion in parsed:
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
