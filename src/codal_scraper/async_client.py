"""
Async client for high-performance Codal API queries

This module provides an asynchronous client for fetching data from
the Codal API with concurrent requests for improved performance.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import aiohttp

from .constants import (
    SEARCH_API_URL, DEFAULT_HEADERS, PERIOD_LENGTHS, COMPANY_TYPES,
    DEFAULT_TIMEOUT, DEFAULT_RETRY_COUNT, DEFAULT_MAX_CONCURRENT
)
from .utils import clean_dict, clean_symbol
from .validators import InputValidator
from .exceptions import APIError, RateLimitError, NetworkError
from .cache import FileCache, CacheConfig
from .rate_limiter import AsyncRateLimiter
from .types import LetterData, QueryParams, QueryStats


logger = logging.getLogger(__name__)


class AsyncCodalClient:
    """
    Async client for high-performance Codal API queries.
    
    Use this client when you need to fetch many pages concurrently
    for significantly improved performance over the synchronous client.
    
    Features:
        - Concurrent page fetching
        - Async rate limiting
        - Response caching
        - Automatic retry with backoff
        - Context manager support
    
    Example:
        >>> async with AsyncCodalClient() as client:
        ...     letters = await client.fetch_board_changes(
        ...         "1402/01/01", "1402/12/29"
        ...     )
        ...     print(f"Found {len(letters)} announcements")
    
    Note:
        Requires aiohttp to be installed: pip install aiohttp
    """
    
    def __init__(
        self,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        timeout: int = DEFAULT_TIMEOUT,
        retry_count: int = DEFAULT_RETRY_COUNT,
        requests_per_second: float = 2.0,
        cache_config: Optional[CacheConfig] = None,
        enable_cache: bool = True
    ):
        """
        Initialize async client.
        
        Args:
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
            retry_count: Number of retry attempts
            requests_per_second: Rate limit for requests
            cache_config: Cache configuration
            enable_cache: Whether to enable caching
        """
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_count = retry_count
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = AsyncRateLimiter(
            requests_per_second=requests_per_second
        )
        
        # Cache setup
        if enable_cache:
            cache_config = cache_config or CacheConfig()
            self.cache = FileCache(cache_config)
        else:
            self.cache = None
        
        # Default query parameters
        self._reset_params()
        
        # Statistics
        self._stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_items': 0,
            'total_time': 0.0
        }
    
    def _reset_params(self) -> None:
        """Reset query parameters to defaults"""
        self.params: QueryParams = {
            "PageNumber": 1,
            "Symbol": -1,
            "CompanyType": -1,
            "LetterCode": -1,
            "FromDate": -1,
            "ToDate": -1,
            "Length": -1,
            "Audited": "true",
            "NotAudited": "true",
            "Consolidatable": "true",
            "NotConsolidatable": "true",
            "Childs": "true",
            "Mains": "true",
        }
    
    async def __aenter__(self) -> 'AsyncCodalClient':
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers=DEFAULT_HEADERS
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit"""
        await self.close()
        return False
    
    async def close(self) -> None:
        """Close the session and release resources"""
        if self._session:
            await self._session.close()
            self._session = None
            logger.debug("Async session closed")
    
    def _build_url(self, params: Dict, page: int = 1) -> str:
        """Build API URL from parameters"""
        params = {**params, 'PageNumber': page}
        cleaned = clean_dict(params)
        query = urlencode(cleaned)
        return f"{SEARCH_API_URL}{query}&search=true"
    
    async def _fetch_page(
        self, 
        url: str, 
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single page with rate limiting and retries.
        
        Args:
            url: URL to fetch
            use_cache: Whether to use cache
            
        Returns:
            Response data or None if failed
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(url)
            if cached is not None:
                self._stats['cache_hits'] += 1
                return cached
            self._stats['cache_misses'] += 1
        
        async with self._semaphore:
            await self.rate_limiter.acquire()
            
            for attempt in range(self.retry_count):
                try:
                    self._stats['requests_made'] += 1
                    
                    async with self._session.get(url) as response:
                        if response.status == 429:
                            retry_after = float(
                                response.headers.get('Retry-After', 60)
                            )
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        
                        response.raise_for_status()
                        data = await response.json()
                        
                        # Cache successful response
                        if use_cache and self.cache and data:
                            self.cache.set(url, data)
                        
                        return data
                        
                except aiohttp.ClientResponseError as e:
                    logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                    self._stats['requests_failed'] += 1
                    
                except aiohttp.ClientError as e:
                    logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                    self._stats['requests_failed'] += 1
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout on attempt {attempt + 1}")
                    self._stats['requests_failed'] += 1
                
                if attempt < self.retry_count - 1:
                    wait_time = min(2 ** attempt, 30)
                    await asyncio.sleep(wait_time)
            
            return None
    
    async def fetch_all_pages(
        self, 
        params: Dict,
        max_pages: Optional[int] = None,
        use_cache: bool = True
    ) -> List[LetterData]:
        """
        Fetch all pages concurrently.
        
        Args:
            params: Query parameters
            max_pages: Maximum pages to fetch (None = all)
            use_cache: Whether to use cache
            
        Returns:
            Combined list of all letters
        """
        start_time = time.time()
        
        # First, get total pages from first request
        first_url = self._build_url(params, page=1)
        first_response = await self._fetch_page(first_url, use_cache)
        
        if not first_response:
            return []
        
        total_pages = first_response.get("Page", 0)
        total_results = first_response.get("Total", 0)
        all_letters: List[LetterData] = first_response.get("Letters", [])
        
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        logger.info(f"Total results: {total_results}, pages: {total_pages}")
        
        if total_pages <= 1:
            self._stats['total_items'] = len(all_letters)
            return all_letters
        
        # Fetch remaining pages concurrently
        tasks = [
            self._fetch_page(self._build_url(params, page=p), use_cache)
            for p in range(2, total_pages + 1)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results, start=2):
            if isinstance(result, dict) and 'Letters' in result:
                all_letters.extend(result['Letters'])
            elif isinstance(result, Exception):
                logger.error(f"Page {i} failed: {result}")
            else:
                logger.warning(f"Page {i} returned no data")
        
        elapsed = time.time() - start_time
        self._stats['total_time'] = elapsed
        self._stats['total_items'] = len(all_letters)
        
        logger.info(
            f"Fetched {len(all_letters)} items from {total_pages} pages "
            f"in {elapsed:.2f}s ({len(all_letters)/elapsed:.1f} items/sec)"
        )
        
        return all_letters
    
    # ============== Convenience Methods ==============
    
    async def fetch_board_changes(
        self,
        from_date: str,
        to_date: str,
        company_type: Optional[str] = None,
        max_pages: Optional[int] = None
    ) -> List[LetterData]:
        """
        Fetch board of directors changes (ن-45).
        
        Args:
            from_date: Start date (Persian calendar)
            to_date: End date (Persian calendar)
            company_type: Optional company type filter
            max_pages: Maximum pages to fetch
            
        Returns:
            List of board change announcements
        """
        InputValidator(from_date).is_date()
        InputValidator(to_date).is_date()
        
        params: QueryParams = {
            "LetterCode": "ن-45",
            "FromDate": from_date,
            "ToDate": to_date,
            "Childs": "false",
            "Mains": "true",
            "Audited": "true",
            "NotAudited": "true",
        }
        
        if company_type:
            params["CompanyType"] = COMPANY_TYPES.get(company_type, company_type)
        
        return await self.fetch_all_pages(params, max_pages)
    
    async def fetch_financial_statements(
        self,
        from_date: str,
        to_date: str,
        period_length: int = 12,
        audited_only: bool = True,
        max_pages: Optional[int] = None
    ) -> List[LetterData]:
        """
        Fetch financial statements (ن-10).
        
        Args:
            from_date: Start date (Persian calendar)
            to_date: End date (Persian calendar)
            period_length: Period length in months
            audited_only: Only fetch audited statements
            max_pages: Maximum pages to fetch
            
        Returns:
            List of financial statement announcements
        """
        InputValidator(from_date).is_date()
        InputValidator(to_date).is_date()
        
        params: QueryParams = {
            "LetterCode": "ن-10",
            "FromDate": from_date,
            "ToDate": to_date,
            "Length": PERIOD_LENGTHS.get(period_length, -1),
        }
        
        if audited_only:
            params["Audited"] = "true"
            params["NotAudited"] = "false"
        
        return await self.fetch_all_pages(params, max_pages)
    
    async def fetch_monthly_reports(
        self,
        from_date: str,
        to_date: str,
        symbol: Optional[str] = None,
        max_pages: Optional[int] = None
    ) -> List[LetterData]:
        """
        Fetch monthly activity reports (ن-30).
        
        Args:
            from_date: Start date (Persian calendar)
            to_date: End date (Persian calendar)
            symbol: Optional specific symbol
            max_pages: Maximum pages to fetch
            
        Returns:
            List of monthly report announcements
        """
        InputValidator(from_date).is_date()
        InputValidator(to_date).is_date()
        
        params: QueryParams = {
            "LetterCode": "ن-30",
            "FromDate": from_date,
            "ToDate": to_date,
        }
        
        if symbol:
            params["Symbol"] = clean_symbol(symbol)
        
        return await self.fetch_all_pages(params, max_pages)
    
    async def fetch_by_symbol(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        max_pages: Optional[int] = None
    ) -> List[LetterData]:
        """
        Fetch all announcements for a specific symbol.
        
        Args:
            symbol: Stock symbol
            from_date: Optional start date
            to_date: Optional end date
            max_pages: Maximum pages to fetch
            
        Returns:
            List of announcements for the symbol
        """
        symbol = clean_symbol(symbol)
        InputValidator(symbol).is_symbol()
        
        params: QueryParams = {
            "Symbol": symbol,
        }
        
        if from_date:
            InputValidator(from_date).is_date()
            params["FromDate"] = from_date
        
        if to_date:
            InputValidator(to_date).is_date()
            params["ToDate"] = to_date
        
        return await self.fetch_all_pages(params, max_pages)
    
    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        from_date: str,
        to_date: str,
        max_pages_per_symbol: int = 10
    ) -> Dict[str, List[LetterData]]:
        """
        Fetch announcements for multiple symbols concurrently.
        
        Args:
            symbols: List of stock symbols
            from_date: Start date (Persian calendar)
            to_date: End date (Persian calendar)
            max_pages_per_symbol: Max pages per symbol
            
        Returns:
            Dictionary mapping symbol to list of announcements
        """
        async def fetch_symbol(symbol: str) -> tuple:
            letters = await self.fetch_by_symbol(
                symbol, from_date, to_date, max_pages_per_symbol
            )
            return symbol, letters
        
        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for result in results:
            if isinstance(result, tuple):
                symbol, letters = result
                output[symbol] = letters
            elif isinstance(result, Exception):
                logger.error(f"Failed to fetch symbol: {result}")
        
        return output
    
    def get_stats(self) -> QueryStats:
        """Get query statistics"""
        return QueryStats(
            total_results=self._stats.get('total_items', 0),
            total_pages=0,  # Not tracked per-query
            pages_fetched=self._stats['requests_made'],
            items_fetched=self._stats['total_items'],
            failed_pages=self._stats['requests_failed'],
            duration_seconds=self._stats['total_time'],
            cache_hits=self._stats['cache_hits'],
            cache_misses=self._stats['cache_misses']
        )
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self._stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_items': 0,
            'total_time': 0.0
        }