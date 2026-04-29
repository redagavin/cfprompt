from unittest.mock import MagicMock

import pandas as pd
import pytest

from cfprompt.study import Study


def _stub_paraphrase_model(scripted: list[str]):
    """Returns canned generations in sequence."""
    m = MagicMock()
    m.cache_id = "para:stub"
    m._scripted = list(scripted)

    def _gen(prompts, per_prompt_seeds=None):
        out = []
        for _ in prompts:
            out.append(m._scripted.pop(0) if m._scripted else "")
        return out

    m.generate = _gen

    class _T:
        def encode(self, t):
            return [hash(w) & 0xFFFF for w in t.split()]

        @property
        def cache_id(self):
            return "tok:para"

    m.tokenizer = _T()
    return m


def _stub_target_model(cache_id: str = "tgt:stub"):
    m = MagicMock()
    m.cache_id = cache_id

    class _T:
        def encode(self, t):
            return [hash(w) & 0xFFFF for w in t.split()]

        @property
        def cache_id(self):
            return "tok:tgt"

    m.tokenizer = _T()
    return m


@pytest.mark.integration
class TestStudyGenerateBaselines:
    def test_basic_run_produces_baselines_df(self):
        df = pd.DataFrame({"q": ["alpha beta gamma delta epsilon zeta eta theta"]})
        # 8-token text; perturbation modifies 1 token (12.5%)
        # paraphrase model returns a candidate with similar edit %
        para = _stub_paraphrase_model(["alpha BETA gamma delta epsilon zeta eta theta"])
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=_stub_target_model(),
            paraphrase_model=para,
            classes=["A", "B"],
            tolerance=5.0,  # generous; one-attempt success
            max_retries=0,
        )
        s.generate_baselines()
        bdf = s._baselines_df
        assert bdf is not None
        assert len(bdf) == 1
        assert "original" in bdf.columns
        assert "target_perturbed" in bdf.columns
        assert "baseline_perturbed" in bdf.columns
        assert "target_edit_pct" in bdf.columns
        assert "baseline_edit_pct" in bdf.columns
        assert "baseline_refused" in bdf.columns
        assert "retries_used" in bdf.columns

    def test_zero_edit_drop(self):
        df = pd.DataFrame({"q": ["alpha beta gamma"], "outcome": ["A"]})
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x,  # identity → zero edit
            prompt_template="{q}",
            target_model=_stub_target_model(),
            paraphrase_model=_stub_paraphrase_model([]),
            classes=["A", "B"],
            tolerance=5.0,
            max_retries=0,
        )
        s.generate_baselines()
        assert len(s._baselines_df) == 0
        assert s._drop_counts["zero_edit"] == 1

    def test_refusal_falls_back_to_original(self):
        df = pd.DataFrame({"q": ["alpha beta gamma delta epsilon"]})
        para = _stub_paraphrase_model(["I can't help with that.", "I cannot assist."])
        s = Study(
            data=df,
            perturb_column="q",
            target_perturbation=lambda x: x.replace("beta", "BETA"),
            prompt_template="{q}",
            target_model=_stub_target_model(),
            paraphrase_model=para,
            classes=["A", "B"],
            tolerance=0.5,
            max_retries=1,
        )
        s.generate_baselines()
        bdf = s._baselines_df
        assert len(bdf) == 1
        assert bool(bdf["baseline_refused"].iloc[0]) is True
        assert bdf["baseline_perturbed"].iloc[0] == bdf["original"].iloc[0]
