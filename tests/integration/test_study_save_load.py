from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfprompt.exceptions import ConfigError, StageNotRunError
from cfprompt.report import Report
from cfprompt.study import Study
from tests.conftest import StubModel as _StubModel


def _build_classification_study():
    df = pd.DataFrame({"q": [f"alpha beta gamma delta {i}" for i in range(5)]})
    rng = np.random.default_rng(0)
    calls = []
    for _ in range(5):
        triple = np.stack(
            [
                rng.dirichlet([2.0, 2.0]),
                rng.dirichlet([1.0, 4.0]),
                rng.dirichlet([2.1, 1.9]),
            ]
        )
        calls.append(triple)
    target = _StubModel(cache_id="tgt:saved", probs_per_call=calls)
    para = _StubModel(
        cache_id="para:saved",
        gens_per_call=[[f"alpha BeTa gamma delta {i}"] for i in range(5)],
    )
    s = Study(
        data=df,
        perturb_column="q",
        target_perturbation=lambda x: x.replace("beta", "BETA"),
        prompt_template="{q}",
        target_model=target,
        paraphrase_model=para,
        classes=["A", "B"],
        tolerance=50.0,
        max_retries=0,
    )
    s.run_all(metrics=["flip_rate"])
    return s


@pytest.mark.integration
class TestStudySaveLoad:
    def test_save_and_load_round_trip_supports_test(self, tmp_path: Path):
        s = _build_classification_study()
        path = tmp_path / "study.pkl"
        s.save(path)

        loaded = Study.load(path)
        report = loaded.test(metrics=["flip_rate"])
        assert isinstance(report, Report)
        assert "loaded_target_cache_id" in report.metadata
        assert report.metadata["loaded_target_cache_id"] == "tgt:saved"
        assert report.metadata["loaded_from_path"].endswith("study.pkl")

    def test_load_no_models_then_run_inference_raises(self, tmp_path: Path):
        s = _build_classification_study()
        path = tmp_path / "study.pkl"
        s.save(path)
        loaded = Study.load(path)
        loaded.reset_inference()
        with pytest.raises(StageNotRunError, match=r"target_model"):
            loaded.run_inference()

    def test_cache_id_mismatch_raises_config_error_on_stage(self, tmp_path: Path):
        s = _build_classification_study()
        path = tmp_path / "study.pkl"
        s.save(path)
        new_target = _StubModel(cache_id="tgt:DIFFERENT")
        new_para = _StubModel(cache_id="para:DIFFERENT")
        loaded = Study.load(
            path,
            target_model=new_target,
            paraphrase_model=new_para,
        )
        loaded.test(metrics=["flip_rate"])
        loaded.reset_inference()
        with pytest.raises(ConfigError, match=r"cache_id"):
            loaded.run_inference()

    def test_paraphrase_only_cache_id_mismatch_raises(self, tmp_path: Path):
        s = _build_classification_study()
        path = tmp_path / "study.pkl"
        s.save(path)
        same_target = _StubModel(cache_id="tgt:saved")
        diff_para = _StubModel(cache_id="para:DIFFERENT")
        loaded = Study.load(
            path,
            target_model=same_target,
            paraphrase_model=diff_para,
        )
        loaded.test(metrics=["flip_rate"])
        loaded.reset_baselines()
        with pytest.raises(ConfigError, match=r"paraphrase_model"):
            loaded.generate_baselines()

    def test_allow_cache_id_mismatch_downgrades_to_warning(self, tmp_path: Path):
        s = _build_classification_study()
        path = tmp_path / "study.pkl"
        s.save(path)
        new_target = _StubModel(cache_id="tgt:DIFFERENT")
        new_para = _StubModel(cache_id="para:DIFFERENT")
        loaded = Study.load(
            path,
            target_model=new_target,
            paraphrase_model=new_para,
            allow_cache_id_mismatch=True,
        )
        loaded.reset_inference()
        loaded.run_inference()

    def test_run_inference_validates_cache_id_when_baselines_preloaded(self, tmp_path: Path):
        """When baselines are already cached and run_inference is called
        with a cache_id mismatch, the validation must still fire instead of
        silently using the loaded baselines."""
        s = _build_classification_study()
        path = tmp_path / "study.pkl"
        s.save(path)
        new_target = _StubModel(cache_id="tgt:DIFFERENT")
        new_para = _StubModel(cache_id="para:saved")  # match para; mismatch tgt
        loaded = Study.load(
            path,
            target_model=new_target,
            paraphrase_model=new_para,
        )
        loaded.reset_inference()
        # baselines_df survives reset_inference; run_inference must still
        # surface the target cache_id mismatch.
        assert loaded._baselines_df is not None
        with pytest.raises(ConfigError, match=r"cache_id"):
            loaded.run_inference()

    def test_freeform_load_without_extract_label_then_run_inference_raises(self, tmp_path: Path):
        """Loading a free-form study without re-supplying extract_label
        leaves the XOR invariant violated. run_inference() must surface
        a StageNotRunError pointing at the fix."""
        df = pd.DataFrame({"q": [f"alpha beta gamma {i}" for i in range(3)]})
        para = _StubModel(
            cache_id="para",
            gens_per_call=[[f"alpha BETA gamma {i}"] for i in range(3)],
        )
        target = _StubModel(
            cache_id="t",
            gens_per_call=[
                ["RESULT: red", "RESULT: red", "RESULT: red"],
                ["RESULT: blue", "RESULT: blue", "RESULT: blue"],
                ["RESULT: green", "RESULT: green", "RESULT: green"],
            ],
        )
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            extract_label=lambda r: "found" if "RESULT" in r else None,
            tolerance=50.0,
            max_retries=0,
        )
        s.generate_baselines()
        path = tmp_path / "study.pkl"
        s.save(path)
        # Load WITHOUT extract_label — XOR invariant violated.
        loaded = Study.load(path)
        assert loaded.mode == "free_form"
        assert loaded.extract_label is None
        loaded.reset_inference()
        with pytest.raises(StageNotRunError, match=r"extract_label"):
            loaded.run_inference()

    def test_reset_inference_allows_rerun(self, tmp_path: Path):
        """reset_inference() must clear the cached frame so a subsequent
        run_inference() recomputes."""
        s = _build_classification_study()
        assert s._inference_df is not None
        original_len = len(s._inference_df)
        s.reset_inference()
        assert s._inference_df is None
        # _baselines_df survives.
        assert s._baselines_df is not None
        assert s._drop_counts["extraction_raised"] == 0
        assert s._drop_counts["extraction_returned_none"] == 0
        # Re-run path: needs fresh probs from the model. Build a fresh stub.
        rng = np.random.default_rng(1)
        triple = np.stack(
            [rng.dirichlet([2.0, 2.0]), rng.dirichlet([1.0, 4.0]), rng.dirichlet([2.1, 1.9])]
        )
        s.target_model._probs = [triple] * original_len
        s.run_inference()
        assert s._inference_df is not None
        assert len(s._inference_df) == original_len

    def test_reset_baselines_clears_inference_too(self):
        """reset_baselines() must clear both baselines and inference, since
        inference depends on baselines."""
        s = _build_classification_study()
        assert s._baselines_df is not None
        assert s._inference_df is not None
        s.reset_baselines()
        assert s._baselines_df is None
        assert s._inference_df is None
        assert s._baseline_refused_count == 0
        assert s._baseline_refused_sample_ids == []

    def test_reextract_throttles_exception_logging(self, caplog):
        """When the new extractor raises on more than 10 samples, only the
        first 10 WARN lines about the raised exception emit, plus a single
        end-of-run summary line."""
        import logging

        n_samples = 15
        df = pd.DataFrame({"q": [f"alpha beta gamma {i}" for i in range(n_samples)]})
        para = _StubModel(
            cache_id="para",
            gens_per_call=[[f"alpha BETA gamma {i}"] for i in range(n_samples)],
        )
        target = _StubModel(
            cache_id="t",
            gens_per_call=[["g0", "g1", "g2"] for _ in range(n_samples)],
        )
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            extract_label=lambda r: r,  # passes; we re-extract with a raising fn
            tolerance=50.0,
            max_retries=0,
        )
        s.generate_baselines()
        s.run_inference()

        def _always_raise(_g):
            raise RuntimeError("boom")

        with caplog.at_level(logging.WARNING, logger="cfprompt"):
            s.reextract(extract_label=_always_raise)
        warn_lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        per_sample_warnings = [m for m in warn_lines if "extract_label raised" in m]
        # 15 samples × 1 raise (loop breaks after first cond) → 15 raises;
        # only 10 logged at WARN per-sample, then 1 summary.
        assert len(per_sample_warnings) == 10
        assert any("more extract_label exceptions suppressed" in m for m in warn_lines)

    def test_reextract_preserves_rows_when_new_extractor_returns_none(self, tmp_path: Path):
        """Non-destructive reextract: when the new extractor returns None
        for a sample, the original label_* values must survive. All input
        rows must remain in inference_df."""
        df = pd.DataFrame({"q": [f"alpha beta gamma {i}" for i in range(3)]})
        para = _StubModel(
            cache_id="para",
            gens_per_call=[[f"alpha BETA gamma {i}"] for i in range(3)],
        )
        target = _StubModel(
            cache_id="t",
            gens_per_call=[
                ["RESULT: red", "RESULT: red", "RESULT: red"],
                ["RESULT: blue", "RESULT: blue", "RESULT: blue"],
                ["nope", "nope", "nope"],  # contains no "RESULT" — first extractor
            ],
        )
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            # First extractor: any non-empty string passes.
            extract_label=lambda r: r.strip() or None,
            tolerance=50.0,
            max_retries=0,
        )
        s.generate_baselines()
        s.run_inference()
        assert len(s._inference_df) == 3
        original_labels = list(s._inference_df["label_orig"])

        # Re-extract with a stricter extractor that returns None on "nope".
        s.reextract(extract_label=lambda r: r.split(": ")[1] if r.startswith("RESULT") else None)
        # All rows survive.
        assert len(s._inference_df) == 3
        new_labels = list(s._inference_df["label_orig"])
        # Rows where the new extractor succeeded got new labels.
        assert new_labels[0] == "red"
        assert new_labels[1] == "blue"
        # Row where the new extractor returned None preserved the old label.
        assert new_labels[2] == original_labels[2]

    def test_reextract_recomputes_labels_without_target_model(self, tmp_path: Path):
        df = pd.DataFrame({"q": [f"alpha beta gamma {i}" for i in range(3)]})
        para = _StubModel(
            cache_id="para",
            gens_per_call=[[f"alpha BETA gamma {i}"] for i in range(3)],
        )
        target = _StubModel(
            cache_id="t",
            gens_per_call=[
                ["RESULT: red", "RESULT: red", "RESULT: red"],
                ["RESULT: blue", "RESULT: blue", "RESULT: blue"],
                ["RESULT: green", "RESULT: green", "RESULT: green"],
            ],
        )
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            extract_label=lambda r: "found" if "RESULT" in r else None,
            tolerance=50.0,
            max_retries=0,
        )
        s.generate_baselines()
        s.run_inference()
        assert len(s._inference_df) == 3
        assert all(s._inference_df["label_orig"] == "found")

        path = tmp_path / "study.pkl"
        s.save(path)
        loaded = Study.load(path)
        loaded.reextract(extract_label=lambda r: r.split()[-1])

        labels = sorted(loaded._inference_df["label_orig"].tolist())
        assert labels == ["blue", "green", "red"]
        assert loaded._drop_counts["extraction_returned_none"] == 0
        assert loaded._drop_counts["extraction_raised"] == 0
