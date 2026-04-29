from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfprompt.exceptions import ConfigError, StageNotRunError
from cfprompt.report import Report
from cfprompt.study import Study


class _StubTokenizer:
    def encode(self, t):
        return [hash(w) & 0xFFFF for w in t.split()]

    @property
    def cache_id(self):
        return "tok:1"


class _StubModel:
    def __init__(self, cache_id="m:1", probs_per_call=None, gens_per_call=None):
        self.cache_id = cache_id
        self.tokenizer = _StubTokenizer()
        self._probs = list(probs_per_call) if probs_per_call else None
        self._gens = list(gens_per_call) if gens_per_call else None

    def score_classes(self, prompts, classes, per_prompt_seeds=None):
        if self._probs:
            return self._probs.pop(0)
        return np.full((len(prompts), len(classes)), 1.0 / len(classes))

    def generate(self, prompts, per_prompt_seeds=None):
        if self._gens:
            return self._gens.pop(0)
        return [""] * len(prompts)

    def close(self):
        pass


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
        loaded._inference_df = None
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
        loaded._inference_df = None
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
        loaded._baselines_df = None
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
        loaded._inference_df = None
        loaded.run_inference()

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
