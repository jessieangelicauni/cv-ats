import tempfile
from pathlib import Path
from src.utils.cache import ExtractionCache


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        cache = ExtractionCache(Path(tmp) / "test.db")
        assert cache.get("nonexistent_key") is None


def test_cache_set_then_get():
    with tempfile.TemporaryDirectory() as tmp:
        cache = ExtractionCache(Path(tmp) / "test.db")
        cache.set("key1", {"role": "engineer", "skills": ["Python"]})
        result = cache.get("key1")
        assert result == {"role": "engineer", "skills": ["Python"]}


def test_cache_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        cache = ExtractionCache(Path(tmp) / "test.db")
        cache.set("key1", {"v": 1})
        cache.set("key1", {"v": 2})
        assert cache.get("key1") == {"v": 2}


def test_make_key_is_deterministic():
    k1 = ExtractionCache.make_key("same content", prefix="jd")
    k2 = ExtractionCache.make_key("same content", prefix="jd")
    assert k1 == k2


def test_make_key_differs_by_prefix():
    k1 = ExtractionCache.make_key("content", prefix="jd")
    k2 = ExtractionCache.make_key("content", prefix="cv")
    assert k1 != k2
