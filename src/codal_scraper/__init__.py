"""
Codal Scraper - Iranian Stock Market Data Extraction System

A comprehensive toolkit for fetching and processing data from Codal.ir,
the official disclosure platform for Iranian listed companies.

Example:
    >>> from codal_scraper import CodalClient, DataProcessor
    >>> 
    >>> with CodalClient() as client:
    ...     letters = client.search_board_changes("1402/01/01", "1402/12/29")
    ...     processor = DataProcessor(letters)
    ...     processor.to_excel("board_changes.xlsx")

For async operations:
    >>> from codal_scraper import AsyncCodalClient
    >>> import asyncio
    >>> 
    >>> async def main():
    ...     async with AsyncCodalClient() as client:
    ...         letters = await client.fetch_board_changes("1402/01/01", "1402/12/29")
    ...     return letters
    >>> 
    >>> data = asyncio.run(main())
"""

__version__ = "1.1.0"
__author__ = "Mohammad Mehdi Pakravan"
__email__ = ""
__license__ = "MIT"

from .exceptions import (
    CodalScraperError,
    ValidationError,
    APIError,
    RateLimitError,
    ScrapingError,
    CacheError,
)
from .types import (
    CompanyType,
    AuditStatus,
    DateRange,
    LetterData,
    BoardMemberData,
    QueryParams,
    ScrapingErrorType,
)
from .constants import (
    YEAR_RANGES,
    LETTER_CODES,
    COMPANY_TYPES,
    PERIOD_LENGTHS,
    get_current_persian_date,
    get_current_persian_year,
)
from .validators import InputValidator, validate_input
from .utils import (
    clean_dict,
    normalize_persian_text,
    persian_to_english_digits,
    datetime_to_num,
    num_to_datetime,
    gregorian_to_shamsi,
    shamsi_to_gregorian,
    clean_symbol,
)
from .cache import FileCache, CacheConfig
from .rate_limiter import RateLimiter, AsyncRateLimiter, RateLimitConfig
from .client import CodalClient
from .processor import DataProcessor
from .board_scraper import BoardMemberScraper

# Async imports (optional - may not be installed)
try:
    from .async_client import AsyncCodalClient
except ImportError:
    AsyncCodalClient = None  # aiohttp not installed

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Exceptions
    "CodalScraperError",
    "ValidationError",
    "APIError",
    "RateLimitError",
    "ScrapingError",
    "CacheError",
    # Types
    "CompanyType",
    "AuditStatus",
    "DateRange",
    "LetterData",
    "BoardMemberData",
    "QueryParams",
    "ScrapingErrorType",
    # Constants
    "YEAR_RANGES",
    "LETTER_CODES",
    "COMPANY_TYPES",
    "PERIOD_LENGTHS",
    "get_current_persian_date",
    "get_current_persian_year",
    # Validators
    "InputValidator",
    "validate_input",
    # Utils
    "clean_dict",
    "normalize_persian_text",
    "persian_to_english_digits",
    "datetime_to_num",
    "clean_symbol",
    # Cache
    "FileCache",
    "CacheConfig",
    # Rate Limiter
    "RateLimiter",
    "AsyncRateLimiter",
    "RateLimitConfig",
    # Clients
    "CodalClient",
    "AsyncCodalClient",
    "DataProcessor",
    "BoardMemberScraper",
]