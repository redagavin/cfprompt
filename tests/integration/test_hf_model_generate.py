# ABOUTME: Integration tests for HFModel.generate (greedy + per-prompt-seed plumbing).
# ABOUTME: Verifies one string per prompt, greedy determinism, and seeds-are-no-op for greedy.
import pytest
import torch

from cfprompt.models.hf import HFModel


@pytest.fixture(scope="module")
def tiny_hf():
    m = HFModel(
        name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        dtype=torch.float32,
        max_new_tokens=8,
    )
    yield m
    m.close()


@pytest.mark.integration
class TestHFModelGenerate:
    def test_generate_returns_one_string_per_prompt(self, tiny_hf):
        out = tiny_hf.generate(["hello", "world", "foo"])
        assert isinstance(out, list)
        assert len(out) == 3
        assert all(isinstance(s, str) for s in out)

    def test_generate_greedy_default_deterministic(self, tiny_hf):
        a = tiny_hf.generate(["hello"])
        b = tiny_hf.generate(["hello"])
        assert a == b

    def test_per_prompt_seeds_recorded_no_op_for_greedy(self, tiny_hf):
        # Greedy decoding ignores seeds; outputs identical regardless.
        a = tiny_hf.generate(["hello"], per_prompt_seeds=[1])
        b = tiny_hf.generate(["hello"], per_prompt_seeds=[2])
        assert a == b
