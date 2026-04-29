import hashlib
import json

import numpy as np
import pytest

from cfprompt.cache import derive_seed


@pytest.mark.unit
class TestDeriveSeed:
    def test_returns_non_negative_int63(self):
        s = derive_seed(42, "abc", 7)
        assert isinstance(s, int)
        assert 0 <= s < (1 << 63)

    def test_deterministic_across_calls(self):
        a = derive_seed(42, "abc", 7)
        b = derive_seed(42, "abc", 7)
        assert a == b

    def test_different_inputs_different_seeds(self):
        a = derive_seed(42, "abc", 7)
        b = derive_seed(42, "abd", 7)
        assert a != b

    def test_different_study_seeds_different_results(self):
        a = derive_seed(42, "abc", 7)
        b = derive_seed(43, "abc", 7)
        assert a != b

    def test_rejects_bytes(self):
        with pytest.raises(TypeError, match="bytes"):
            derive_seed(42, b"abc")

    def test_rejects_numpy_int(self):
        with pytest.raises(TypeError):
            derive_seed(42, np.int64(7))

    def test_rejects_dict(self):
        with pytest.raises(TypeError):
            derive_seed(42, {"a": 1})

    def test_accepts_tuple_of_primitives(self):
        s = derive_seed(42, ("abc", 7, "def"))
        assert isinstance(s, int)

    def test_matches_expected_sha256_blob(self):
        # Hand-computed reference: SHA-256 of canonical JSON of [42, "abc", 7]
        expected_blob = json.dumps([42, "abc", 7], sort_keys=True).encode("utf-8")
        expected_raw = int.from_bytes(hashlib.sha256(expected_blob).digest()[:8], "big")
        expected = expected_raw & ((1 << 63) - 1)
        assert derive_seed(42, "abc", 7) == expected
