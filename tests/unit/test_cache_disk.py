from pathlib import Path

import pytest

from cfprompt.cache import DiskCache


@pytest.mark.unit
class TestDiskCache:
    def test_set_and_get_roundtrip(self, tmp_path: Path):
        c = DiskCache(tmp_path, namespace="paraphrase")
        c.set("a" * 64, {"x": 1, "y": [1, 2, 3]})
        assert c.get("a" * 64) == {"x": 1, "y": [1, 2, 3]}

    def test_missing_key_returns_default(self, tmp_path: Path):
        c = DiskCache(tmp_path, namespace="inference")
        assert c.get("b" * 64) is None
        assert c.get("b" * 64, default="MISSING") == "MISSING"

    def test_uses_hash_prefix_sharding(self, tmp_path: Path):
        c = DiskCache(tmp_path, namespace="paraphrase")
        key = "abcd" + "0" * 60
        c.set(key, "value")
        expected = tmp_path / "paraphrase" / "ab" / "cd" / f"{key}.pkl"
        assert expected.exists()

    def test_atomic_write_uses_tmp_then_rename(self, tmp_path: Path):
        c = DiskCache(tmp_path, namespace="x")
        c.set("e" * 64, "v")
        leftovers = list(tmp_path.rglob("*.tmp"))
        assert leftovers == []

    def test_namespace_isolation(self, tmp_path: Path):
        a = DiskCache(tmp_path, namespace="paraphrase")
        b = DiskCache(tmp_path, namespace="inference")
        a.set("k" * 64, "from_a")
        assert b.get("k" * 64) is None

    def test_overwrite_replaces_value(self, tmp_path: Path):
        c = DiskCache(tmp_path, namespace="x")
        c.set("k" * 64, "v1")
        c.set("k" * 64, "v2")
        assert c.get("k" * 64) == "v2"
