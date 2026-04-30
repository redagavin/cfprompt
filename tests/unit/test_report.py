import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfprompt.report import Report, TestResult, _json_safe


def _make_test_result(metric: str = "jsd", **kwargs) -> TestResult:
    base = dict(
        metric=metric,
        test="paired_t",
        statistic=1.5,
        p_value=0.05,
        p_value_kind="two-sided",
        ci_low=0.1,
        ci_high=0.3,
        ci_kind="asymptotic",
        n=100,
        extra={"degenerate": False, "mean_diff": 0.2, "std_diff": 1.0},
    )
    base.update(kwargs)
    return TestResult(**base)


@pytest.mark.unit
class TestReport:
    def test_summary_table_columns(self):
        r = Report(
            results=[_make_test_result("jsd"), _make_test_result("flip_rate")],
            metadata={"n_input": 200, "drop_counts": {}, "cfprompt_version": "0.0.1"},
        )
        df = r.summary_table()
        assert "metric" in df.columns
        assert "p_value" in df.columns
        assert "p_value_kind" in df.columns
        assert "n_dropped" in df.columns
        assert "degenerate" in df.columns
        assert "bootstrap_unstable" in df.columns
        assert "n_dropped_resamples" in df.columns
        assert len(df) == 2

    def test_to_json_roundtrip(self, tmp_path: Path):
        r = Report(
            results=[_make_test_result()],
            metadata={"n_input": 100, "cfprompt_version": "0.0.1"},
        )
        path = tmp_path / "out.json"
        r.to_json(path)
        loaded = json.loads(path.read_text())
        assert "results" in loaded
        assert "metadata" in loaded
        assert loaded["metadata"]["n_input"] == 100

    def test_to_excel_writes_sheets(self, tmp_path: Path):
        pytest.importorskip("openpyxl")
        r = Report(
            results=[_make_test_result()],
            metadata={
                "n_input": 100,
                "cfprompt_version": "0.0.1",
                "drop_counts": {"zero_edit": 3, "extraction_returned_none": 1},
                "baseline_refused_count": 2,
                "baseline_refused_sample_ids": [10, 20],
            },
        )
        path = tmp_path / "out.xlsx"
        r.to_excel(path)
        assert path.exists()
        sheets = pd.ExcelFile(path).sheet_names
        assert "Results" in sheets
        assert "Metadata" in sheets
        assert "Drop Counts" in sheets

    def test_merge_concatenates_results(self):
        a = Report(
            results=[_make_test_result("jsd")],
            metadata={"n_input": 100, "cfprompt_version": "0.0.1"},
        )
        b = Report(
            results=[_make_test_result("flip_rate")],
            metadata={"n_input": 100, "cfprompt_version": "0.0.1"},
        )
        merged = a.merge(b)
        assert len(merged.results) == 2
        assert "merged_from" in merged.metadata
        assert len(merged.metadata["merged_from"]) == 2

    def test_merge_flattens_nested_merged_from(self):
        """Repeated merge must produce a flat list of leaf metadatas, not a
        nested chain - otherwise downstream readers must recurse."""
        a = Report(results=[_make_test_result("jsd")], metadata={"src": "a"})
        b = Report(results=[_make_test_result("flip_rate")], metadata={"src": "b"})
        c = Report(results=[_make_test_result("kl")], metadata={"src": "c"})
        merged = a.merge(b).merge(c)
        assert len(merged.results) == 3
        leaves = merged.metadata["merged_from"]
        assert isinstance(leaves, list)
        assert len(leaves) == 3
        for leaf in leaves:
            assert "merged_from" not in leaf
        assert {leaf["src"] for leaf in leaves} == {"a", "b", "c"}

    def test_merge_warns_on_divergent_metadata(self, caplog):
        a = Report(
            results=[_make_test_result("jsd")],
            metadata={"cfprompt_version": "0.0.1", "n_input": 100},
        )
        b = Report(
            results=[_make_test_result("flip_rate")],
            metadata={"cfprompt_version": "0.0.2", "n_input": 100},
        )
        with caplog.at_level(logging.WARNING, logger="cfprompt.report"):
            a.merge(b)
        assert any(
            "cfprompt_version" in rec.message and "differ" in rec.message
            for rec in caplog.records
        )


@pytest.mark.unit
class TestJsonSafe:
    def test_nan_becomes_null(self):
        assert _json_safe(float("nan")) is None

    def test_inf_becomes_null(self):
        assert _json_safe(float("inf")) is None
        assert _json_safe(float("-inf")) is None

    def test_finite_floats_pass_through(self):
        assert _json_safe(1.5) == 1.5
        assert _json_safe(0.0) == 0.0

    def test_numpy_int_converted_to_int(self):
        v = _json_safe(np.int64(42))
        assert v == 42 and isinstance(v, int)

    def test_numpy_float_converted_to_float(self):
        v = _json_safe(np.float64(3.14))
        assert isinstance(v, float)
        assert math.isclose(v, 3.14)

    def test_numpy_float_nan_becomes_null(self):
        assert _json_safe(np.float64("nan")) is None

    def test_numpy_ndarray_converted_to_list(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        assert _json_safe(arr) == [1, 2, 3]

    def test_bool_preserved(self):
        assert _json_safe(True) is True
        assert _json_safe(False) is False

    def test_to_json_with_nan_statistic_produces_strict_parseable_json(
        self, tmp_path: Path
    ):
        """RFC 8259 strict parsers reject literal `NaN`. The Report must emit
        valid JSON even when a TestResult has non-finite statistics."""
        r = Report(
            results=[_make_test_result(statistic=float("nan"), p_value=float("inf"))],
            metadata={"n_input": 100, "rate": float("nan")},
        )
        path = tmp_path / "out.json"
        r.to_json(path)
        text = path.read_text()
        assert "NaN" not in text
        assert "Infinity" not in text
        loaded = json.loads(text)  # strict parse
        assert loaded["results"][0]["statistic"] is None
        assert loaded["results"][0]["p_value"] is None
        assert loaded["metadata"]["rate"] is None

    def test_to_json_with_numpy_types_roundtrips(self, tmp_path: Path):
        r = Report(
            results=[_make_test_result()],
            metadata={"n": np.int64(5), "vals": np.array([1.0, 2.0])},
        )
        path = tmp_path / "out.json"
        r.to_json(path)
        loaded = json.loads(path.read_text())
        assert loaded["metadata"]["n"] == 5
        assert loaded["metadata"]["vals"] == [1.0, 2.0]
