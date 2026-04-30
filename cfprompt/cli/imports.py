"""Shared helper for resolving ``module:attr`` strings to Python objects."""

from __future__ import annotations

import importlib
from typing import Any


def resolve_callable(spec: str) -> Any:
    """Resolve ``module:attr`` (with optional dotted attribute path) to an object.

    Examples:
        resolve_callable("os.path:join")
        resolve_callable("collections:OrderedDict.fromkeys")
    """
    if ":" not in spec:
        raise ValueError(f"expected module:function, got {spec!r}")
    mod_name, attr_path = spec.split(":", 1)
    obj = importlib.import_module(mod_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj
