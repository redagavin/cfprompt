"""TestResult dataclass + Report container with serialization."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd


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
            "results": [asdict(r) for r in self.results],
            "metadata": _json_safe(self.metadata),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))

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
        per-source metadata preserved under metadata['merged_from']."""
        new_meta = dict(self.metadata)
        new_meta["merged_from"] = [dict(self.metadata), dict(other.metadata)]
        return Report(results=self.results + other.results, metadata=new_meta)

    def __repr__(self) -> str:
        df = self.summary_table()
        return f"Report({len(self.results)} results)\n{df.to_string(index=False)}"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
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
