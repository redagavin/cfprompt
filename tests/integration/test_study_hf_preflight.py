# ABOUTME: Verifies Study.__init__ fail-fasts on HFModel class-prefix collisions
# ABOUTME: instead of deferring the error to run_inference time.
import pandas as pd
import pytest
import torch

from cfprompt.exceptions import ClassificationModeError
from cfprompt.models.hf import HFModel
from cfprompt.study import Study


# Use class_prefix="" to make the test resilient across transformers versions.
# Older transformers tokenize " yes"/" no" as [29871, ...] (collision on the
# leading-space token), while newer versions don't. With class_prefix="" and
# classes ["yessir", "yep"] both starting with the SentencePiece token "▁y",
# the collision is guaranteed for any Llama-style tokenizer regardless of
# version (matches the assertion in tests/integration/test_hf_model_score.py).
@pytest.fixture(scope="module")
def hf_no_prefix():
    m = HFModel(
        name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        dtype=torch.float32,
        class_prefix="",
    )
    yield m
    m.close()


def _stub_paraphrase():
    class _T:
        def encode(self, t):
            return [hash(w) & 0xFFFF for w in t.split()]

        @property
        def cache_id(self):
            return "tok:para"

    class _Para:
        cache_id = "para:stub"
        tokenizer = _T()

        def generate(self, prompts, per_prompt_seeds=None):
            return [""] * len(prompts)

        def close(self):
            pass

    return _Para()


@pytest.mark.integration
class TestStudyHFPreflight:
    def test_collision_raises_at_construction(self, hf_no_prefix):
        df = pd.DataFrame({"q": ["alpha beta gamma"]})
        with pytest.raises(ClassificationModeError, match="first token"):
            Study(
                data=df,
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="{q}",
                target_model=hf_no_prefix,
                paraphrase_model=_stub_paraphrase(),
                classes=["yessir", "yep"],
            )

    def test_error_message_names_the_colliding_classes(self, hf_no_prefix):
        df = pd.DataFrame({"q": ["alpha beta gamma"]})
        with pytest.raises(ClassificationModeError) as exc_info:
            Study(
                data=df,
                perturb_column="q",
                target_perturbation=lambda x: x.upper(),
                prompt_template="{q}",
                target_model=hf_no_prefix,
                paraphrase_model=_stub_paraphrase(),
                classes=["yessir", "yep"],
            )
        msg = str(exc_info.value)
        assert "yessir" in msg
        assert "yep" in msg
