"""
Caching layer for Codal Scraper

This module provides file-based caching for API responses to reduce
redundant requests and improve performance.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import CacheError


logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """
    Configuration for the cache system.
    
    Attributes:
        cache_dir: Directory to store cache files
        default_ttl: Default time-to-live in seconds
        max_size_mb: Maximum cache size in megabytes
        enabled: Whether caching is enabled
        cleanup_on_init: Whether to cleanup expired entries on initialization
    """
    cache_dir: str = ".codal_cache"
    default_ttl: int = 3600  # 1 hour
    max_size_mb: int = 100
    enabled: bool = True
    cleanup_on_init: bool = True


class FileCache:
    """
    Simple file-based cache for API responses.
    
    This cache stores responses as JSON files with metadata about
    expiration times. It's suitable for caching API responses that
    don't change frequently.
    
    Example:
        >>> cache = FileCache()
        >>> cache.set("https://api.example.com/data", {"key": "value"})
        >>> data = cache.get("https://api.example.com/data")
        >>> print(data)
        {'key': 'value'}
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the file cache.
        
        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self._stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'errors': 0
        }
        
        if self.config.enabled:
            self._initialize_cache_dir()
            if self.config.cleanup_on_init:
                self.cleanup()
    
    def _initialize_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create cache directory: {e}")
            self.config.enabled = False
    
    def _get_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """
        Generate a unique cache key from URL and params.
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            MD5 hash as cache key
        """
        key_data = url
        if params:
            # Sort params for consistent hashing
            sorted_params = json.dumps(params, sort_keys=True)
            key_data += sorted_params
        
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for a cache key"""
        return self.cache_dir / f"{key}.json"
    
    def get(
        self, 
        url: str, 
        params: Optional[Dict] = None
    ) -> Optional[Any]:
        """
        Get cached value if it exists and is not expired.
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            Cached data or None if not found/expired
        """
        if not self.config.enabled:
            return None
        
        key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self._stats['misses'] += 1
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check expiration
            if time.time() > cached.get('expires_at', 0):
                self._delete_cache_file(cache_path)
                self._stats['misses'] += 1
                return None
            
            self._stats['hits'] += 1
            logger.debug(f"Cache hit for {url}")
            return cached.get('data')
            
        except json.JSONDecodeError as e:
            logger.warning(f"Cache file corrupted: {cache_path} - {e}")
            self._delete_cache_file(cache_path)
            self._stats['errors'] += 1
            return None
        except IOError as e:
            logger.warning(f"Cache read error: {e}")
            self._stats['errors'] += 1
            return None
    
    def set(
        self, 
        url: str, 
        data: Any, 
        params: Optional[Dict] = None, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store value in cache.
        
        Args:
            url: Request URL
            data: Data to cache
            params: Request parameters
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config.enabled:
            return False
        
        key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(key)
        ttl = ttl or self.config.default_ttl
        
        try:
            cache_data = {
                'url': url,
                'params': params,
                'data': data,
                'created_at': time.time(),
                'expires_at': time.time() + ttl
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=None)
            
            self._stats['writes'] += 1
            logger.debug(f"Cached {url}")
            return True
            
        except (IOError, TypeError) as e:
            logger.warning(f"Cache write error: {e}")
            self._stats['errors'] += 1
            return False
    
    def invalidate(
        self, 
        url: Optional[str] = None, 
        params: Optional[Dict] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            url: Specific URL to invalidate (if None, clears all cache)
            params: Request parameters for specific entry
            
        Returns:
            Number of entries invalidated
        """
        if not self.config.enabled:
            return 0
        
        count = 0
        
        if url:
            key = self._get_cache_key(url, params)
            cache_path = self._get_cache_path(key)
            if self._delete_cache_file(cache_path):
                count = 1
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.json"):
                if self._delete_cache_file(cache_file):
                    count += 1
        
        logger.info(f"Invalidated {count} cache entries")
        return count
    
    def cleanup(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        if not self.config.enabled:
            return 0
        
        now = time.time()
        removed = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                
                if now > cached.get('expires_at', 0):
                    if self._delete_cache_file(cache_file):
                        removed += 1
                        
            except (json.JSONDecodeError, IOError):
                # Remove corrupted files
                if self._delete_cache_file(cache_file):
                    removed += 1
        
        if removed > 0:
            logger.info(f"Cache cleanup: removed {removed} expired entries")
        
        return removed
    
    def _delete_cache_file(self, path: Path) -> bool:
        """Safely delete a cache file"""
        try:
            path.unlink()
            return True
        except OSError:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.config.enabled:
            return {'enabled': False}
        
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        hit_rate = 0
        total_requests = self._stats['hits'] + self._stats['misses']
        if total_requests > 0:
            hit_rate = self._stats['hits'] / total_requests * 100
        
        return {
            'enabled': True,
            'entries': len(cache_files),
            'size_mb': round(total_size / 1024 / 1024, 2),
            'max_size_mb': self.config.max_size_mb,
            'cache_dir': str(self.cache_dir),
            'default_ttl': self.config.default_ttl,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'writes': self._stats['writes'],
            'errors': self._stats['errors'],
            'hit_rate_percent': round(hit_rate, 2)
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics"""
        self._stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'errors': 0
        }
    
    def is_cached(self, url: str, params: Optional[Dict] = None) -> bool:
        """
        Check if a URL is cached without retrieving data.
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            True if cached and not expired
        """
        if not self.config.enabled:
            return False
        
        key = self._get_cache_key(url, params)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            return time.time() <= cached.get('expires_at', 0)
        except:
            return False
    
    def get_or_set(
        self,
        url: str,
        fetch_func: callable,
        params: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get from cache or fetch and cache.
        
        This is a convenience method that combines get and set operations.
        
        Args:
            url: Request URL
            fetch_func: Callable that returns data to cache
            params: Request parameters
            ttl: Time-to-live in seconds
            
        Returns:
            Cached or freshly fetched data
        """
        cached = self.get(url, params)
        if cached is not None:
            return cached
        
        data = fetch_func()
        if data is not None:
            self.set(url, data, params, ttl)
        
        return data


class MemoryCache:
    """
    Simple in-memory cache for session-level caching.
    
    This cache is faster than FileCache but doesn't persist
    between program runs.
    """
    
    def __init__(self, max_entries: int = 1000, default_ttl: int = 300):
        """
        Initialize memory cache.
        
        Args:
            max_entries: Maximum number of entries to store
            default_ttl: Default time-to-live in seconds
        """
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._stats = {'hits': 0, 'misses': 0}
    
    def _get_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key"""
        key_data = url
        if params:
            key_data += json.dumps(params, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached value"""
        key = self._get_key(url, params)
        entry = self._cache.get(key)
        
        if entry is None:
            self._stats['misses'] += 1
            return None
        
        if time.time() > entry['expires_at']:
            del self._cache[key]
            self._stats['misses'] += 1
            return None
        
        self._stats['hits'] += 1
        return entry['data']
    
    def set(
        self, 
        url: str, 
        data: Any, 
        params: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> None:
        """Set cached value"""
        # Evict oldest entries if at capacity
        if len(self._cache) >= self.max_entries:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k]['created_at']
            )
            del self._cache[oldest_key]
        
        key = self._get_key(url, params)
        ttl = ttl or self.default_ttl
        
        self._cache[key] = {
            'data': data,
            'created_at': time.time(),
            'expires_at': time.time() + ttl
        }
    
    def clear(self) -> None:
        """Clear all cached entries"""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total * 100 if total > 0 else 0
        
        return {
            'entries': len(self._cache),
            'max_entries': self.max_entries,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate_percent': round(hit_rate, 2)
        }