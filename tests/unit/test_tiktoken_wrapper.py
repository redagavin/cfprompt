# ABOUTME: Unit tests for TiktokenWrapper - verifies Tokenizer Protocol conformance.
# ABOUTME: Covers encode return type and cache_id format.
import pytest

from cfprompt.models.base import Tokenizer
from cfprompt.models.openai import TiktokenWrapper


@pytest.mark.unit
class TestTiktokenWrapper:
    def test_implements_protocol(self):
        t = TiktokenWrapper("o200k_base")
        assert isinstance(t, Tokenizer)

    def test_encode_returns_list_of_int(self):
        t = TiktokenWrapper("o200k_base")
        ids = t.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_cache_id_format(self):
        t = TiktokenWrapper("o200k_base")
        assert t.cache_id == "tiktoken:o200k_base"
