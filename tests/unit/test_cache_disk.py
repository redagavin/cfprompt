import logging
from pathlib import Path
from unittest import mock

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

    def test_path_rejects_non_64_char_key(self, tmp_path: Path):
        c = DiskCache(tmp_path, namespace="x")
        with pytest.raises(ValueError, match="64-char sha256"):
            c._path("abc")
        with pytest.raises(ValueError, match="64-char sha256"):
            c._path("a" * 63)
        with pytest.raises(ValueError, match="64-char sha256"):
            c._path("a" * 65)

    def test_corrupt_cache_logs_warning_and_returns_default(self, tmp_path: Path, caplog):
        """Garbage bytes at the cache path must not crash get(); after the
        retry exhausts, a WARNING is logged and `default` is returned."""
        c = DiskCache(tmp_path, namespace="x")
        key = "c" * 64
        path = c._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"this is not a pickle stream")

        sentinel = object()
        with caplog.at_level(logging.WARNING, logger="cfprompt.cache"):
            result = c.get(key, default=sentinel)

        assert result is sentinel
        assert any(
            "corrupt" in rec.message.lower() and str(path) in rec.message for rec in caplog.records
        )

    def test_corrupt_cache_handles_struct_error(self, tmp_path: Path, caplog):
        """A truncated pickle that raises struct.error must also be treated
        as a corrupt-cache miss, not propagate the exception."""
        c = DiskCache(tmp_path, namespace="x")
        key = "d" * 64
        path = c._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pickle protocol 4 framing prefix that's structurally incomplete.
        path.write_bytes(b"\x80\x04\x95")

        with caplog.at_level(logging.WARNING, logger="cfprompt.cache"):
            assert c.get(key, default="MISS") == "MISS"
        assert any("corrupt" in rec.message.lower() for rec in caplog.records)

    def test_set_fsyncs_before_replace(self, tmp_path: Path):
        """Durability guarantee: set() must flush + fsync the tmp file BEFORE
        the atomic replace, so a crash post-rename cannot expose half-written
        bytes as a valid cache entry."""
        c = DiskCache(tmp_path, namespace="x")
        order: list[str] = []

        real_fsync = __import__("os").fsync
        real_replace = __import__("os").replace

        def tracking_fsync(fd):
            order.append("fsync")
            return real_fsync(fd)

        def tracking_replace(src, dst):
            order.append("replace")
            return real_replace(src, dst)

        with (
            mock.patch("cfprompt.cache.os.fsync", side_effect=tracking_fsync),
            mock.patch("cfprompt.cache.os.replace", side_effect=tracking_replace),
        ):
            c.set("f" * 64, "value")

        assert order == ["fsync", "replace"]
