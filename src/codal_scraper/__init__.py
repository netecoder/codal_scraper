"""
Codal Scraper - Iranian Stock Market Data Extraction System
A comprehensive toolkit for fetching and processing data from Codal.ir
"""

__version__ = "1.0.0"
__author__ = "Mohammad Mehdi Pakravan"
__all__ = [
    "CodalClient",
    "DataProcessor",
    "BoardMemberScraper",
    "validate_input",
    "YEAR_RANGES",
    "LETTER_CODES"
]

from .client import CodalClient
from .processor import DataProcessor
from .board_scraper import BoardMemberScraper
from .validators import validate_input
from .constants import YEAR_RANGES, LETTER_CODES