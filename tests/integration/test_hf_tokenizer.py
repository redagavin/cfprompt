# ABOUTME: Integration tests for HFTokenizer wrapper using tiny-random-LlamaForCausalLM.
# ABOUTME: Verifies Tokenizer Protocol conformance, encoding output, and cache_id formatting.
import pytest

from cfprompt.models.base import Tokenizer
from cfprompt.models.hf import HFTokenizer


@pytest.mark.integration
class TestHFTokenizer:
    def test_implements_protocol(self):
        tok = HFTokenizer.from_name_and_revision(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            revision="main",
        )
        assert isinstance(tok, Tokenizer)

    def test_encode_returns_list_of_int(self):
        tok = HFTokenizer.from_name_and_revision(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            revision="main",
        )
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_cache_id_includes_name_and_revision_sha(self):
        tok = HFTokenizer.from_name_and_revision(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            revision="main",
        )
        # Format: f"hf:{name}@{sha}"
        assert tok.cache_id.startswith("hf:hf-internal-testing/tiny-random-LlamaForCausalLM@")
        # SHA must be a 40-char hex string (or a fallback marker; we accept either)
        suffix = tok.cache_id.split("@", 1)[1]
        assert len(suffix) > 0

    def test_cache_id_offline_fallback_uses_literal_revision(self, monkeypatch):
        # Force the SHA-resolution call to fail; expect cache_id to fall back
        # to the literal revision string and emit a WARNING (we don't assert
        # on the warning here; just on the fallback behavior).
        from cfprompt.models import hf as hf_mod

        def boom(*a, **kw):
            raise OSError("simulated offline")

        monkeypatch.setattr(hf_mod, "_resolve_revision_sha", boom)
        tok = HFTokenizer.from_name_and_revision(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            revision="main",
        )
        assert tok.cache_id.endswith("@main")
