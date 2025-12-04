"""
Rate limiting utilities for Codal Scraper

This module provides rate limiting to avoid overwhelming the Codal API
and to comply with rate limits.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting.
    
    Attributes:
        requests_per_second: Maximum requests per second
        burst_limit: Maximum burst of requests allowed
        retry_after_429: Seconds to wait after receiving 429 response
    """
    requests_per_second: float = 2.0
    burst_limit: int = 5
    retry_after_429: float = 60.0


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.
    
    This rate limiter allows bursts of requests up to the burst_limit,
    while maintaining an average rate of requests_per_second.
    
    Example:
        >>> limiter = RateLimiter()
        >>> for url in urls:
        ...     limiter.acquire()  # Blocks if rate limit exceeded
        ...     response = requests.get(url)
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._tokens = float(self.config.burst_limit)
        self._last_update = time.monotonic()
        self._lock = Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self._last_update
        
        # Add tokens based on time passed
        self._tokens = min(
            float(self.config.burst_limit),
            self._tokens + elapsed * self.config.requests_per_second
        )
        self._last_update = now
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token for making a request.
        
        Args:
            blocking: If True, wait for a token; if False, return immediately
            timeout: Maximum time to wait for a token (None = wait forever)
            
        Returns:
            True if token acquired, False if not (when non-blocking or timeout)
        """
        start_time = time.monotonic()
        
        while True:
            with self._lock:
                self._refill()
                
                if self._tokens >= 1:
                    self._tokens -= 1
                    return True
                
                if not blocking:
                    return False
                
                # Calculate wait time
                wait_time = (1 - self._tokens) / self.config.requests_per_second
            
            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)
            
            time.sleep(wait_time)
    
    def get_wait_time(self) -> float:
        """
        Get estimated wait time for next request.
        
        Returns:
            Seconds until a token will be available
        """
        with self._lock:
            self._refill()
            
            if self._tokens >= 1:
                return 0.0
            
            return (1 - self._tokens) / self.config.requests_per_second
    
    def reset(self) -> None:
        """Reset rate limiter to initial state"""
        with self._lock:
            self._tokens = float(self.config.burst_limit)
            self._last_update = time.monotonic()
    
    @property
    def available_tokens(self) -> float:
        """Get number of available tokens"""
        with self._lock:
            self._refill()
            return self._tokens


class AsyncRateLimiter:
    """
    Async-native rate limiter for use with asyncio.
    
    Example:
        >>> limiter = AsyncRateLimiter()
        >>> async def fetch(url):
        ...     await limiter.acquire()
        ...     async with session.get(url) as response:
        ...         return await response.json()
    """
    
    def __init__(
        self, 
        requests_per_second: float = 2.0,
        burst_limit: int = 5
    ):
        """
        Initialize async rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
            burst_limit: Maximum burst of requests
        """
        self.requests_per_second = requests_per_second
        self.burst_limit = burst_limit
        self.min_interval = 1.0 / requests_per_second
        
        self._tokens = float(burst_limit)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire a token for making a request.
        
        This coroutine will wait if necessary to maintain the rate limit.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            
            # Refill tokens
            self._tokens = min(
                float(self.burst_limit),
                self._tokens + elapsed * self.requests_per_second
            )
            self._last_update = now
            
            if self._tokens >= 1:
                self._tokens -= 1
                return
            
            # Calculate wait time
            wait_time = (1 - self._tokens) / self.requests_per_second
        
        # Wait outside the lock
        await asyncio.sleep(wait_time)
        
        async with self._lock:
            self._tokens = max(0, self._tokens - 1)
    
    async def get_wait_time(self) -> float:
        """Get estimated wait time for next request"""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            
            tokens = min(
                float(self.burst_limit),
                self._tokens + elapsed * self.requests_per_second
            )
            
            if tokens >= 1:
                return 0.0
            
            return (1 - tokens) / self.requests_per_second
    
    def reset(self) -> None:
        """Reset rate limiter"""
        self._tokens = float(self.burst_limit)
        self._last_update = time.monotonic()


class AdaptiveRateLimiter:
    """
    Rate limiter that adapts based on server responses.
    
    This limiter automatically adjusts its rate when it detects
    rate limiting responses (429 status codes) or when requests
    succeed consistently.
    """
    
    def __init__(
        self,
        initial_rate: float = 2.0,
        min_rate: float = 0.1,
        max_rate: float = 10.0,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
        success_threshold: int = 10
    ):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_rate: Starting requests per second
            min_rate: Minimum requests per second
            max_rate: Maximum requests per second
            increase_factor: Factor to increase rate on success
            decrease_factor: Factor to decrease rate on 429
            success_threshold: Consecutive successes before increasing rate
        """
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.success_threshold = success_threshold
        
        self._current_rate = initial_rate
        self._consecutive_successes = 0
        self._limiter = RateLimiter(RateLimitConfig(
            requests_per_second=initial_rate
        ))
        self._lock = Lock()
    
    def acquire(self) -> bool:
        """Acquire a token"""
        return self._limiter.acquire()
    
    def record_success(self) -> None:
        """Record a successful request"""
        with self._lock:
            self._consecutive_successes += 1
            
            if self._consecutive_successes >= self.success_threshold:
                self._increase_rate()
                self._consecutive_successes = 0
    
    def record_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """Record a rate limit response"""
        with self._lock:
            self._consecutive_successes = 0
            self._decrease_rate()
            
            if retry_after:
                logger.info(f"Rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
    
    def _increase_rate(self) -> None:
        """Increase the rate limit"""
        new_rate = min(
            self._current_rate * self.increase_factor,
            self.max_rate
        )
        
        if new_rate != self._current_rate:
            logger.debug(f"Increasing rate from {self._current_rate:.2f} to {new_rate:.2f}")
            self._current_rate = new_rate
            self._update_limiter()
    
    def _decrease_rate(self) -> None:
        """Decrease the rate limit"""
        new_rate = max(
            self._current_rate * self.decrease_factor,
            self.min_rate
        )
        
        if new_rate != self._current_rate:
            logger.info(f"Decreasing rate from {self._current_rate:.2f} to {new_rate:.2f}")
            self._current_rate = new_rate
            self._update_limiter()
    
    def _update_limiter(self) -> None:
        """Update the underlying rate limiter"""
        self._limiter = RateLimiter(RateLimitConfig(
            requests_per_second=self._current_rate
        ))
    
    @property
    def current_rate(self) -> float:
        """Get current rate limit"""
        return self._current_rate