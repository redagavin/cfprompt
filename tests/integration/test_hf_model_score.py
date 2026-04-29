# ABOUTME: Integration tests for HFModel.score_classes (greedy logits at last position).
# ABOUTME: Verifies output shape, row normalization, column ordering, and collision detection.
import numpy as np
import pytest
import torch

from cfprompt.exceptions import ClassificationModeError
from cfprompt.models.hf import HFModel


@pytest.fixture(scope="module")
def tiny_hf():
    # Use class_prefix="" because Llama's SentencePiece tokenizer encodes any
    # leading-space string as [29871, ...], which would make every class
    # collide on the leading-space token under the default class_prefix=" ".
    m = HFModel(
        name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        dtype=torch.float32,
        class_prefix="",
    )
    yield m
    m.close()


@pytest.mark.integration
class TestHFModelScoreClasses:
    def test_returns_correct_shape(self, tiny_hf):
        prompts = ["Q: hello\nA:", "Q: world\nA:"]
        classes = ["yes", "no"]
        probs = tiny_hf.score_classes(prompts, classes)
        assert probs.shape == (2, 2)

    def test_rows_sum_to_one(self, tiny_hf):
        prompts = ["Q: hello\nA:", "Q: world\nA:"]
        classes = ["yes", "no", "maybe"]
        probs = tiny_hf.score_classes(prompts, classes)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_columns_aligned_to_user_class_order(self, tiny_hf):
        # Same prompts but reversed classes — softmax over the same two token
        # ids will produce mirrored columns.
        prompts = ["Q: hi\nA:"]
        a = tiny_hf.score_classes(prompts, ["yes", "no"])
        b = tiny_hf.score_classes(prompts, ["no", "yes"])
        np.testing.assert_allclose(a, b[:, ::-1], atol=1e-6)

    def test_first_token_collision_raises(self, tiny_hf):
        # Two classes whose first-token id collides under the Llama
        # SentencePiece tokenizer with class_prefix="": both "yessir" and
        # "yep" start with the BPE token "_y" (id 343), so first-token
        # scoring would conflate them.
        prompts = ["Q:\nA:"]
        with pytest.raises(ClassificationModeError, match="first token"):
            tiny_hf.score_classes(prompts, ["yessir", "yep"])
