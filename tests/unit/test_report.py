import json
from pathlib import Path

import pandas as pd
import pytest

from cfprompt.report import Report, TestResult


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
