"""TestResult dataclass + Report container.

This file will grow significantly in Phase 8. The current minimal version
exists to support stats.py tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TestResult:
    metric: str
    test: str
    statistic: float | None
    p_value: float
    p_value_kind: Literal["one-sided", "two-sided"]
    p_value_two_sided: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_kind: Literal["percentile", "asymptotic"] | None = None
    effect_size: float | None = None
    n: int = 0
    extra: dict = field(default_factory=dict)
