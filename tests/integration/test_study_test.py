import numpy as np
import pandas as pd
import pytest

from cfprompt.exceptions import CfpromptError, ConfigError
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


@pytest.mark.integration
class TestStudyTest:
    def _classification_study(self):
        df = pd.DataFrame(
            {"q": [f"alpha beta gamma delta epsilon zeta eta theta {i}" for i in range(20)]}
        )
        rng = np.random.default_rng(0)
        probs_orig = rng.dirichlet([2.0, 2.0], size=20)
        probs_target = rng.dirichlet([1.0, 4.0], size=20)
        probs_base = rng.dirichlet([2.1, 1.9], size=20)
        calls = []
        for i in range(20):
            calls.append(np.stack([probs_orig[i], probs_target[i], probs_base[i]]))
        para = _StubModel(
            cache_id="para",
            gens_per_call=[
                [f"alpha BeTa gamma delta epsilon zeta eta theta {i}"] for i in range(20)
            ],
        )
        target = _StubModel(cache_id="tgt", probs_per_call=calls)
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
            n_bootstrap=200,
        )
        return s

    def test_classification_jsd_returns_report(self):
        s = self._classification_study()
        report = s.run_all(metrics=["jsd"])
        assert isinstance(report, Report)
        assert len(report.results) == 1
        r = report.results[0]
        assert r.metric == "jsd"
        assert r.test == "paired_t"
        assert r.p_value_kind == "two-sided"
        assert r.statistic is not None
        assert 0.0 <= r.p_value <= 1.0

    def test_metric_mode_incompatibility_raises_at_test_time(self):
        df = pd.DataFrame({"q": ["alpha beta gamma delta epsilon"]})
        target = _StubModel(cache_id="t", gens_per_call=[["x", "y", "z"]])
        para = _StubModel(cache_id="p", gens_per_call=[["alpha BETA gamma delta epsilon"]])
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            extract_label=lambda r: r,
            tolerance=50.0,
            max_retries=0,
        )
        s.generate_baselines()
        s.run_inference()
        with pytest.raises(ConfigError, match=r"jsd.*classification mode"):
            s.test(metrics=["jsd"])

    def test_run_all_chains_stages(self):
        s = self._classification_study()
        assert s._baselines_df is None
        assert s._inference_df is None
        report = s.run_all(metrics=["flip_rate"])
        assert s._baselines_df is not None
        assert s._inference_df is not None
        assert len(report.results) == 1
        assert report.results[0].metric == "flip_rate"

    def test_regression_model_kwarg_ignored_when_no_regression_metric(self):
        s = self._classification_study()
        report = s.run_all(metrics=["jsd"], regression_model="level")
        assert report.metadata.get("regression_model") in (None, "level")

    def test_n_zero_freeform_error_includes_example_generations(self):
        """When every free-form sample fails extraction, the error message
        must include up to 3 truncated example raw generations to help the
        user fix their extract_label."""
        df = pd.DataFrame({"q": [f"alpha beta gamma delta epsilon {i}" for i in range(2)]})
        # Long generation that will be truncated (>200 chars).
        long_gen = "X" * 250
        target = _StubModel(
            cache_id="t",
            gens_per_call=[
                [long_gen, long_gen, long_gen],
                ["short fail", "short fail", "short fail"],
            ],
        )
        para = _StubModel(
            cache_id="p",
            gens_per_call=[[f"alpha BeTa gamma delta epsilon {i}"] for i in range(2)],
        )
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            # Always returns None — every sample dropped.
            extract_label=lambda r: None,
            tolerance=50.0,
            max_retries=0,
        )
        s.generate_baselines()
        s.run_inference()
        assert len(s._inference_df) == 0
        with pytest.raises(CfpromptError) as exc_info:
            s.test(metrics=["flip_rate"])
        msg = str(exc_info.value)
        assert "Example raw generations" in msg
        # Long generation truncated and given an ellipsis suffix.
        assert "…" in msg
        # Short generation surfaced verbatim.
        assert "short fail" in msg

    def test_regression_outcome_class_lookup_raises_cfprompt_error(self):
        """If inference_df has outcome_class values not in self.classes (e.g.
        the user mutated the frame after preflight), _run_regression must
        raise CfpromptError, not the raw ValueError from list.index."""
        df = pd.DataFrame(
            {
                "q": [f"alpha beta gamma delta epsilon zeta eta theta {i}" for i in range(20)],
                "dir": [1, -1] * 10,
                "outcome": ["A"] * 20,
            }
        )
        rng = np.random.default_rng(0)
        calls = [
            np.stack(
                [
                    rng.dirichlet([2.0, 2.0]),
                    rng.dirichlet([1.0, 4.0]),
                    rng.dirichlet([2.1, 1.9]),
                ]
            )
            for _ in range(20)
        ]
        para = _StubModel(
            cache_id="para",
            gens_per_call=[
                [f"alpha BeTa gamma delta epsilon zeta eta theta {i}"] for i in range(20)
            ],
        )
        target = _StubModel(cache_id="tgt", probs_per_call=calls)
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=target,
            paraphrase_model=para,
            classes=["A", "B"],
            direction_column="dir",
            outcome_class_column="outcome",
            alternative="greater",
            tolerance=50.0,
            max_retries=0,
            n_bootstrap=200,
        )
        s.generate_baselines()
        s.run_inference()
        # Mutate the inference frame to slip an invalid value past preflight.
        s._inference_df.loc[s._inference_df.index[0], "outcome"] = "Z"
        with pytest.raises(CfpromptError, match=r"outcome_class_column"):
            s.test(metrics=["regression"], regression_model="difference")
