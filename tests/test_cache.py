import time

from codal_scraper.cache import CacheConfig, FileCache


def test_file_cache_roundtrip(tmp_path):
    config = CacheConfig(cache_dir=str(tmp_path / "cache"), cleanup_on_init=False)
    cache = FileCache(config)
    url = "https://example.com/api"
    assert cache.get(url) is None
    cache.set(url, {"value": 1})
    assert cache.get(url)["value"] == 1


def test_cache_respects_ttl(tmp_path):
    config = CacheConfig(cache_dir=str(tmp_path / "cache_ttl"), default_ttl=1, cleanup_on_init=False)
    cache = FileCache(config)
    url = "https://example.com/ttl"
    cache.set(url, {"value": 42})
    assert cache.get(url)["value"] == 42
    time.sleep(1.1)
    assert cache.get(url) is None
