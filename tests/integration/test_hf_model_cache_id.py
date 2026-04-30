# ABOUTME: Integration tests for HFModel construction, cache_id, and close().
# ABOUTME: Verifies dtype/class_prefix appear in cache_id and close()/context-manager work.
import pytest
import torch

from cfprompt.models.hf import HFModel


@pytest.mark.integration
class TestHFModelCacheId:
    def test_cache_id_includes_name_revision_dtype_class_prefix(self):
        m = HFModel(
            name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            dtype=torch.float32,
            class_prefix=" ",
        )
        cid = m.cache_id
        # Format: hf:{name}@{sha}|dtype={dtype}|class_prefix={repr(class_prefix)}|max_new_tokens={n}
        assert cid.startswith("hf:hf-internal-testing/tiny-random-LlamaForCausalLM@")
        assert "|dtype=torch.float32" in cid
        assert "|class_prefix=' '" in cid
        assert "|max_new_tokens=" in cid
        m.close()

    def test_cache_id_changes_with_max_new_tokens(self):
        """Different max_new_tokens must produce distinct cache_ids so the
        inference cache is invalidated when the user changes generation
        length."""
        m1 = HFModel(
            name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            dtype=torch.float32,
            max_new_tokens=64,
        )
        m2 = HFModel(
            name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            dtype=torch.float32,
            max_new_tokens=128,
        )
        try:
            assert m1.cache_id != m2.cache_id
            assert "|max_new_tokens=64" in m1.cache_id
            assert "|max_new_tokens=128" in m2.cache_id
        finally:
            m1.close()
            m2.close()

    def test_cache_id_changes_with_class_prefix(self):
        """Regression: different class_prefix must produce distinct cache_ids
        so paraphrase-vs-baseline scoring caches don't bleed together when
        the user re-runs with a different prefix."""
        m1 = HFModel(
            name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            dtype=torch.float32,
            class_prefix=" ",
        )
        m2 = HFModel(
            name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            dtype=torch.float32,
            class_prefix="",
        )
        try:
            assert m1.cache_id != m2.cache_id
            assert "|class_prefix=' '" in m1.cache_id
            assert "|class_prefix=''" in m2.cache_id
        finally:
            m1.close()
            m2.close()

    def test_close_idempotent(self):
        m = HFModel(name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM")
        m.close()
        m.close()  # second call must not raise

    def test_context_manager_closes(self):
        with HFModel(name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM") as m:
            assert m._model is not None
        # After __exit__ the model reference is cleared
        assert m._model is None
