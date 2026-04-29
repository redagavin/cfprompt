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
        # f"hf:{name}@{sha-or-label}|dtype={dtype}|class_prefix={repr(class_prefix)}"
        assert cid.startswith("hf:hf-internal-testing/tiny-random-LlamaForCausalLM@")
        assert "|dtype=torch.float32" in cid
        assert "|class_prefix=' '" in cid
        m.close()

    def test_close_idempotent(self):
        m = HFModel(name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM")
        m.close()
        m.close()  # second call must not raise

    def test_context_manager_closes(self):
        with HFModel(name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM") as m:
            assert m._model is not None
        # After __exit__ the model reference is cleared
        assert m._model is None
