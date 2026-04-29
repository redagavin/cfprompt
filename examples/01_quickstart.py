"""Minimal end-to-end cfprompt example using a tiny random Llama (CPU only).

Run: python examples/01_quickstart.py
"""
from unittest.mock import MagicMock

import pandas as pd
import torch

from cfprompt.models.hf import HFModel
from cfprompt.study import Study


def upper_perturbation(text: str) -> str:
    return text.upper()


def main() -> None:
    df = pd.DataFrame({"q": [f"Question number {i} about topic Z." for i in range(20)]})

    target_model = HFModel(
        name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        dtype=torch.float32,
        # SentencePiece tokenizers (Llama 1/2) collapse a leading space to a
        # single token, which collides across all classes; class_prefix=""
        # avoids that. Newer Llama 3 (BPE) does not need this override.
        class_prefix="",
    )
    para = MagicMock()
    para.cache_id = "para:stub"
    para.generate = lambda prompts, per_prompt_seeds=None: [
        f"Reworded #{i}: question Z." for i in range(len(prompts))
    ]

    study = Study(
        data=df,
        perturb_column="q",
        target_perturbation=upper_perturbation,
        prompt_template="Q: {q}\nA:",
        target_model=target_model,
        paraphrase_model=para,
        classes=["yes", "no"],
        tolerance=50.0,
        max_retries=0,
    )
    report = study.run_all(metrics=["flip_rate"])
    print(report.summary_table().to_string(index=False))
    target_model.close()


if __name__ == "__main__":
    main()
