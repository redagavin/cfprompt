"""TestResult dataclass + Report container with serialization.

JSON output contract: NaN and ±Infinity float values are sanitized to ``null``
so emitted JSON conforms to RFC 8259 (the Python default ``allow_nan=True``
emits literal ``NaN``/``Infinity`` tokens, which strict parsers reject).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


class Report:
    """Container for one or more TestResults plus run-level metadata."""

    def __init__(self, results: list[TestResult], metadata: dict | None = None) -> None:
        self.results = list(results)
        self.metadata = dict(metadata or {})

    def summary_table(self) -> pd.DataFrame:
        n_input = self.metadata.get("n_input", 0)
        rows = []
        for r in self.results:
            row = {
                "metric": r.metric,
                "test": r.test,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "p_value_kind": r.p_value_kind,
                "p_value_two_sided": r.p_value_two_sided,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "ci_kind": r.ci_kind,
                "n": r.n,
                "n_dropped": n_input - r.n if n_input else 0,
                "degenerate": r.extra.get("degenerate", False),
                "n_dropped_resamples": r.extra.get("n_dropped_resamples", 0),
                "bootstrap_unstable": r.extra.get("bootstrap_unstable", False),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "results": [_json_safe(asdict(r)) for r in self.results],
            "metadata": _json_safe(self.metadata),
        }
        path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False))

    def to_excel(self, path: str | Path) -> None:
        path = Path(path)
        with pd.ExcelWriter(path) as xw:
            self.summary_table().to_excel(xw, sheet_name="Results", index=False)
            _flatten_metadata(self.metadata).to_excel(xw, sheet_name="Metadata", index=False)
            _drop_counts_table(self.metadata).to_excel(xw, sheet_name="Drop Counts", index=False)
            refused_ids = self.metadata.get("baseline_refused_sample_ids") or []
            if refused_ids:
                pd.DataFrame({"sample_id": refused_ids}).to_excel(
                    xw, sheet_name="Refused Samples", index=False
                )

    def merge(self, other: Report) -> Report:
        """Concatenate `results` lists. Top-level metadata is from self;
        per-source metadata preserved under metadata['merged_from'] as a
        flat list of leaf metadatas (no nested merged_from chains).

        Logs a WARNING listing keys whose values differ between the two
        sources (excluding ``merged_from``); merge proceeds regardless.
        """
        self_leaf = {k: v for k, v in self.metadata.items() if k != "merged_from"}
        other_leaf = {k: v for k, v in other.metadata.items() if k != "merged_from"}

        divergent = sorted(
            k for k in set(self_leaf) & set(other_leaf)
            if self_leaf[k] != other_leaf[k]
        )
        if divergent:
            logger.warning(
                "Report.merge: metadata values differ between sources for keys: %s",
                divergent,
            )

        existing = self.metadata.get("merged_from")
        if isinstance(existing, list):
            merged_from = list(existing)
        else:
            merged_from = [self_leaf]
        other_existing = other.metadata.get("merged_from")
        if isinstance(other_existing, list):
            merged_from.extend(other_existing)
        else:
            merged_from.append(other_leaf)

        new_meta = dict(self.metadata)
        new_meta["merged_from"] = merged_from
        return Report(results=self.results + other.results, metadata=new_meta)

    def __repr__(self) -> str:
        df = self.summary_table()
        return f"Report({len(self.results)} results)\n{df.to_string(index=False)}"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    # bool must be checked before int (bool is a subclass of int)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    return str(obj)


def _flatten_metadata(metadata: dict) -> pd.DataFrame:
    """Flatten metadata into a 2-column key/value DataFrame."""
    items = []
    for k, v in metadata.items():
        items.append({"key": k, "value": json.dumps(_json_safe(v), default=str)})
    return pd.DataFrame(items)


def _drop_counts_table(metadata: dict) -> pd.DataFrame:
    drops = metadata.get("drop_counts", {}) or {}
    n_input = metadata.get("n_input", 0)
    rows = []
    for k, v in drops.items():
        rate = (v / n_input) if n_input else 0.0
        rows.append({"reason": k, "count": v, "rate": rate})
    if not rows:
        return pd.DataFrame(columns=["reason", "count", "rate"])
    return pd.DataFrame(rows)
