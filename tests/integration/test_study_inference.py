import numpy as np
import pandas as pd
import pytest

from cfprompt.study import Study
from tests.conftest import StubModel as _StubModel
from tests.conftest import StubTokenizer as _StubTokenizer


@pytest.mark.integration
class TestStudyRunInference:
    def test_classification_records_probs_per_condition(self):
        df = pd.DataFrame({"q": ["alpha beta gamma delta epsilon"]})
        para = _StubModel(
            cache_id="para:1",
            gens_per_call=[["alpha BETA gamma delta epsilon"]],
        )
        target = _StubModel(
            cache_id="tgt:1",
            probs_per_call=[np.array([[0.6, 0.4], [0.5, 0.5], [0.55, 0.45]])],
        )
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            classes=["A", "B"],
            tolerance=5.0,
            max_retries=0,
        )
        s.generate_baselines()
        s.run_inference()
        idf = s._inference_df
        assert len(idf) == 1
        assert "probs_orig" in idf.columns
        assert "probs_target" in idf.columns
        assert "probs_base" in idf.columns
        np.testing.assert_allclose(idf["probs_orig"].iloc[0], [0.6, 0.4])
        np.testing.assert_allclose(idf["probs_target"].iloc[0], [0.5, 0.5])
        np.testing.assert_allclose(idf["probs_base"].iloc[0], [0.55, 0.45])

    def test_free_form_extraction_failures(self):
        df = pd.DataFrame({"q": ["one two three four five six"]})
        para = _StubModel(
            cache_id="p:1",
            gens_per_call=[["one two THREE four five six"]],
        )
        target = _StubModel(
            cache_id="t:1",
            gens_per_call=[["I won't answer", "Yes", "No"]],
        )

        def extract(r):
            if r.startswith("I won't"):
                return None
            return r

        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("three", "THREE"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            extract_label=extract,
            tolerance=5.0,
            max_retries=0,
        )
        s.generate_baselines()
        s.run_inference()
        # Original returned None → entire sample dropped.
        assert len(s._inference_df) == 0
        assert s._drop_counts["extraction_returned_none"] == 1
        assert s._drop_counts["extraction_raised"] == 0

    def test_cache_dir_second_run_skips_model_calls(self, tmp_path):
        """First Study.run_all() populates the cache; second run with the
        same cache_dir does NOT invoke the paraphrase model or target
        model. Verifies the §6.6 cache integration end-to-end."""
        df = pd.DataFrame({"q": [f"alpha beta gamma delta {i}" for i in range(5)]})

        class _CountingModel:
            def __init__(self, cache_id, *, probs=None, gens=None):
                self.cache_id = cache_id
                self.tokenizer = _StubTokenizer()
                self._probs = probs
                self._gens = gens
                self.score_calls = 0
                self.generate_calls = 0

            def score_classes(self, prompts, classes, per_prompt_seeds=None):
                self.score_calls += 1
                k = len(classes)
                if self._probs is not None:
                    return np.full((len(prompts), k), self._probs)
                return np.full((len(prompts), k), 1.0 / k)

            def generate(self, prompts, per_prompt_seeds=None):
                self.generate_calls += 1
                return self._gens.pop(0) if self._gens else [""] * len(prompts)

            def close(self):
                pass

        para_first = _CountingModel(
            "para:cached",
            gens=[[f"alpha BeTa gamma delta {i}"] for i in range(5)],
        )
        target_first = _CountingModel("tgt:cached", probs=[0.6, 0.4])

        s1 = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target_first,
            paraphrase_model=para_first,
            classes=["A", "B"],
            tolerance=50.0,
            max_retries=0,
            cache_dir=tmp_path,
            seed=42,
        )
        # NOTE: plan calls run_all(); we use generate_baselines+run_inference
        # because run_all is added in Task 10.4 (chicken-and-egg fix).
        s1.generate_baselines()
        s1.run_inference()
        first_para_calls = para_first.generate_calls
        first_score_calls = target_first.score_calls
        assert first_para_calls > 0
        assert first_score_calls > 0

        para_second = _CountingModel("para:cached")
        target_second = _CountingModel("tgt:cached", probs=[0.6, 0.4])
        s2 = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target_second,
            paraphrase_model=para_second,
            classes=["A", "B"],
            tolerance=50.0,
            max_retries=0,
            cache_dir=tmp_path,
            seed=42,
        )
        s2.generate_baselines()
        s2.run_inference()
        assert para_second.generate_calls == 0
        assert target_second.score_calls == 0

    def test_free_form_extraction_raises_counts_separately(self):
        # A buggy extractor that raises on every sample must NOT be silently
        # masked as `extraction_returned_none`. The split counters surface
        # the failure mode at default log level.
        df = pd.DataFrame({"q": ["one two three four"]})
        para = _StubModel(
            cache_id="p:1",
            gens_per_call=[["one two THREE four"]],
        )
        target = _StubModel(
            cache_id="t:1",
            gens_per_call=[["any", "any", "any"]],
        )

        def buggy_extractor(r):
            raise ValueError(f"my regex broke on: {r!r}")

        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("three", "THREE"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            extract_label=buggy_extractor,
            tolerance=5.0,
            max_retries=0,
        )
        s.generate_baselines()
        s.run_inference()
        assert len(s._inference_df) == 0
        assert s._drop_counts["extraction_returned_none"] == 0
        assert s._drop_counts["extraction_raised"] == 1

    def test_run_inference_throttles_extract_label_exception_logging(self, caplog):
        """When extract_label raises on more than 10 free-form samples, only
        the first 10 WARN lines emit, then a single end-of-run summary."""
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

        def _always_raise(_g):
            raise ValueError("boom")

        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            extract_label=_always_raise,
            tolerance=50.0,
            max_retries=0,
        )
        s.generate_baselines()
        with caplog.at_level(logging.WARNING, logger="cfprompt"):
            s.run_inference()
        warn_lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        per_sample = [m for m in warn_lines if "extract_label raised" in m]
        # Inference loop breaks after first failing condition per sample, so
        # n_extraction_raised == n_samples == 15. First 10 logged, then summary.
        assert len(per_sample) == 10
        assert any("more extract_label exceptions suppressed" in m for m in warn_lines)
        assert s._drop_counts["extraction_raised"] == n_samples

    def test_openai_missing_class_drops_sample(self):
        """When score_classes returns NaN for any condition (mimicking OpenAI
        returning logprobs that don't include the requested class), the sample
        must be dropped and counted under openai_missing_class."""
        df = pd.DataFrame({"q": ["alpha beta gamma delta"]})
        # First call returns NaN in target arm
        calls = [np.array([[0.6, 0.4], [np.nan, np.nan], [0.55, 0.45]])]
        target = _StubModel(cache_id="tgt", probs_per_call=calls)
        para = _StubModel(cache_id="p", gens_per_call=[["alpha BETA gamma delta"]])
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
        s.generate_baselines()
        s.run_inference()
        assert len(s._inference_df) == 0
        assert s._drop_counts["openai_missing_class"] == 1
        assert s._drop_counts["zero_edit"] == 0
        assert s._drop_counts["tokenization_failed"] == 0
        assert s._drop_counts["extraction_returned_none"] == 0
        assert s._drop_counts["extraction_raised"] == 0

    def test_tokenization_failed_drops_sample(self):
        """When the target_model.tokenizer.encode raises, the sample is
        dropped and counted under tokenization_failed."""

        class _RaisingTokenizer:
            def encode(self, t):
                raise ValueError("tokenizer broke")

            @property
            def cache_id(self):
                return "tok:raise"

        class _RaisingModel:
            cache_id = "tgt:raise"
            tokenizer = _RaisingTokenizer()

            def score_classes(self, prompts, classes, per_prompt_seeds=None):
                return np.full((len(prompts), len(classes)), 0.5)

            def generate(self, prompts, per_prompt_seeds=None):
                return [""] * len(prompts)

            def close(self):
                pass

        df = pd.DataFrame({"q": ["alpha beta gamma"]})
        para = _StubModel(cache_id="p", gens_per_call=[["alpha BETA gamma"]])
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=_RaisingModel(),
            paraphrase_model=para,
            classes=["A", "B"],
            tolerance=50.0,
            max_retries=0,
        )
        s.generate_baselines()
        assert len(s._baselines_df) == 0
        assert s._drop_counts["tokenization_failed"] == 1
        assert s._drop_counts["openai_missing_class"] == 0
        assert s._drop_counts["zero_edit"] == 0
        assert s._drop_counts["extraction_returned_none"] == 0
        assert s._drop_counts["extraction_raised"] == 0
