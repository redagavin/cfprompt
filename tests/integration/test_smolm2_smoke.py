"""End-to-end smoke test using SmolLM2-135M (small real model).

Marked as @pytest.mark.smoke so default pytest runs skip it. Run explicitly via
`pytest -m smoke`. Downloads SmolLM2-135M (~270MB) on first run.
"""

import pandas as pd
import pytest
import torch

from cfprompt.models.hf import HFModel


class _SmokeParaphraseModel:
    """Concrete paraphrase stub - returns canned outputs trimmed to len(prompts).

    Using a real class instead of MagicMock surfaces missing-method bugs
    (e.g. if Study starts calling para.tokenizer or para.close()).
    """

    cache_id = "para:smoke"
    _outputs = [
        "Sky's color: blue?",
        "Wetness of water?",
        "Hot fire, yes/no?",
    ]

    def generate(self, prompts, per_prompt_seeds=None):
        return self._outputs[: len(prompts)]


@pytest.mark.smoke
class TestSmolM2Smoke:
    def test_classification_run_completes(self):
        m = HFModel(
            name_or_path="HuggingFaceTB/SmolLM2-135M",
            dtype=torch.float32,
        )
        try:
            df = pd.DataFrame(
                {
                    "q": [
                        "Is the sky blue?",
                        "Is water wet?",
                        "Is fire hot?",
                    ]
                }
            )
            para = _SmokeParaphraseModel()
            from cfprompt.study import Study

            s = Study(
                data=df,
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="Q: {q}\nA (yes/no):",
                target_model=m,
                paraphrase_model=para,
                classes=["yes", "no"],
                tolerance=50.0,
                max_retries=0,
            )
            report = s.run_all(metrics=["flip_rate"])
            assert len(report.results) == 1
            r = report.results[0]
            assert 0.0 <= r.statistic <= 1.0
        finally:
            m.close()
