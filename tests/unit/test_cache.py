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


from cfprompt.cache import safe_format
from cfprompt.exceptions import ConfigError


@pytest.mark.unit
class TestSafeFormat:
    def test_simple_substitution(self):
        out = safe_format("hello {name}", {"name": "world"})
        assert out == "hello world"

    def test_multiple_placeholders(self):
        out = safe_format("Q: {q}\nA: {a}", {"q": "1+1", "a": "2"})
        assert out == "Q: 1+1\nA: 2"

    def test_literal_braces_pass_through(self):
        out = safe_format("set: {{x, y}} and {name}", {"name": "z"})
        assert out == "set: {x, y} and z"

    def test_value_with_braces_not_recursed(self):
        # A column value containing {foo} must not be re-interpreted as a placeholder
        out = safe_format("text: {col}", {"col": "this has {foo} and {bar}"})
        assert out == "text: this has {foo} and {bar}"

    def test_value_with_unbalanced_braces_passes_through(self):
        out = safe_format("{col}", {"col": "weird }} text"})
        assert out == "weird }} text"

    def test_rejects_attribute_access(self):
        with pytest.raises(ConfigError, match="attribute"):
            safe_format("{x.foo}", {"x": "bar"})

    def test_rejects_indexing(self):
        with pytest.raises(ConfigError, match=r"\[|index"):
            safe_format("{x[0]}", {"x": ["a"]})

    def test_rejects_conversion(self):
        with pytest.raises(ConfigError, match=r"conversion|!"):
            safe_format("{x!r}", {"x": "y"})

    def test_rejects_format_spec(self):
        with pytest.raises(ConfigError, match=r"format spec|:"):
            safe_format("{x:>10}", {"x": "y"})

    def test_rejects_nested_format_spec(self):
        with pytest.raises(ConfigError, match=r"format spec|nested"):
            safe_format("{x:{y}}", {"x": 1, "y": "d"})

    def test_rejects_positional_placeholder(self):
        with pytest.raises(ConfigError, match=r"positional|empty"):
            safe_format("{}", {})

    def test_rejects_numeric_placeholder(self):
        with pytest.raises(ConfigError, match=r"positional|number"):
            safe_format("{0}", {})

    def test_missing_key_raises_config_error(self):
        with pytest.raises(ConfigError, match=r"options|column"):
            safe_format("{question} -> {options}", {"question": "Q"})

    def test_missing_key_message_lists_columns(self):
        with pytest.raises(ConfigError) as exc:
            safe_format("{a} {b}", {"a": "1"})
        msg = str(exc.value)
        assert "b" in msg
        # Available columns should be surfaced
        assert "a" in msg
