"""
Utility functions for data processing and conversion
"""

import re
from typing import Any, Dict, Optional, Union
from jdatetime import date as jd
from jdatetime import datetime as jdt
import pandas as pd
import numpy as np

from .constants import FA_TO_EN_DIGITS, AR_TO_FA_LETTER


def clean_dict(dictionary: dict) -> dict:
    """Remove None and -1 values from dictionary"""
    return {k: v for k, v in dictionary.items() if v not in [None, -1]}


def normalize_persian_text(text: str) -> str:
    """Normalize Persian/Arabic characters"""
    if not text:
        return ''
    
    # Convert Arabic to Persian letters
    for ar, fa in AR_TO_FA_LETTER.items():
        text = text.replace(ar, fa)
    
    # Remove zero-width characters
    text = text.replace('\u200c', '').replace('\u200f', '').replace('\u200e', '')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def persian_to_english_digits(text: str) -> str:
    """Convert Persian/Arabic digits to English"""
    if not text:
        return text
    
    for fa, en in FA_TO_EN_DIGITS.items():
        text = text.replace(fa, en)
    
    return text


def datetime_to_num(dt: Union[str, None]) -> Optional[int]:
    """Convert datetime string to numeric format"""
    if not dt or dt == "":
        return None
    
    try:
        # Remove non-numeric characters
        dt = re.sub(r'[^0-9]', '', str(dt))
        
        # Handle empty string after cleaning
        if not dt:
            return None
            
        # Pad with zeros to get 14 digits (YYYYMMDDHHmmss)
        dt_len = len(dt)
        if dt_len > 14:
            dt = dt[:14]  # Truncate if too long
        elif dt_len < 14:
            dt = dt.ljust(14, '0')  # Pad with zeros if too short
            
        return int(dt)
    except (ValueError, TypeError) as e:
        print(f"Error converting datetime: {e}")
        return None


def num_to_datetime(num: Union[int, str], 
                   datetime: bool = True, 
                   date_sep: str = "/", 
                   time_sep: str = ":", 
                   dt_sep: str = " ") -> str:
    """
    Convert numeric datetime to string format
    
    Args:
        num: Numeric datetime (YYYYMMDDHHmmss)
        datetime: If True, include time component
        date_sep: Separator for date components
        time_sep: Separator for time components
        dt_sep: Separator between date and time
    
    Returns:
        Formatted datetime string
    """
    num = str(num).zfill(14)  # Ensure 14 digits
    
    date_part = f"{num[0:4]}{date_sep}{num[4:6]}{date_sep}{num[6:8]}"
    
    if datetime:
        time_part = f"{num[8:10]}{time_sep}{num[10:12]}{time_sep}{num[12:14]}"
        return f"{date_part}{dt_sep}{time_part}"
    
    return date_part


def gregorian_to_shamsi(date: Union[str, int]) -> str:
    """Convert Gregorian date (YYYYMMDD) to Shamsi (Persian) calendar"""
    date = str(date)
    
    if len(date) != 8:
        raise ValueError(f"Date must be in YYYYMMDD format, got: {date}")
    
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    
    shamsi_date = jd.fromgregorian(day=day, month=month, year=year)
    return shamsi_date.strftime("%Y/%m/%d")


def shamsi_to_gregorian(date: str) -> str:
    """Convert Shamsi (Persian) date to Gregorian (YYYYMMDD)"""
    # Parse the Shamsi date
    date_obj = jdt.strptime(date.replace('/', ''), "%Y%m%d")
    
    # Convert to Gregorian
    greg_date = date_obj.togregorian()
    
    return greg_date.strftime("%Y%m%d")


def value_to_float(value: Union[str, int, float]) -> float:
    """
    Convert formatted values to float
    Handles K (thousands), M (millions), B (billions) suffixes
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return 0.0
    
    # Remove commas
    value = value.replace(',', '')
    
    # Handle suffixes
    multipliers = {
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
        'T': 1_000_000_000_000
    }
    
    for suffix, multiplier in multipliers.items():
        if suffix in value:
            try:
                number = float(value.replace(suffix, '').strip())
                return number * multiplier
            except ValueError:
                return 0.0
    
    try:
        return float(value)
    except ValueError:
        return 0.0


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case"""
    # Insert underscore before uppercase letters
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    
    return name.lower()


def dataframe_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all DataFrame column names to snake_case"""
    df.columns = [to_snake_case(col) for col in df.columns]
    return df


def calculate_date_range(year: int) -> tuple:
    """
    Calculate the start and end dates for a Persian calendar year
    
    Args:
        year: Persian calendar year
    
    Returns:
        Tuple of (start_date, end_date) in YYYY/MM/DD format
    """
    start_date = f"{year}/01/01"
    end_date = f"{year}/12/29"  # Persian calendar months can have 29, 30, or 31 days
    
    return start_date, end_date


def parse_codal_response(response: dict) -> pd.DataFrame:
    """
    Parse Codal API response into a DataFrame
    
    Args:
        response: API response dictionary
    
    Returns:
        DataFrame with parsed data
    """
    if not response or 'Letters' not in response:
        return pd.DataFrame()
    
    letters = response['Letters']
    
    if not letters:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(letters)
    
    # Convert column names to snake_case
    df = dataframe_columns_to_snake_case(df)
    
    # Parse dates if present
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df


def clean_symbol(symbol: str) -> str:
    """Clean and normalize stock symbol"""
    if not symbol:
        return ""
    
    # Remove whitespace
    symbol = symbol.strip()
    
    # Convert Persian digits to English
    symbol = persian_to_english_digits(symbol)
    
    # Remove special characters (keep only alphanumeric and Persian letters)
    symbol = re.sub(r'[^\w\u0600-\u06FF]+', '', symbol)
    
    return symbol.upper()


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def safe_get(dictionary: dict, *keys, default=None):
    """
    Safely get nested dictionary values
    
    Args:
        dictionary: The dictionary to search
        *keys: Keys to traverse
        default: Default value if key not found
    
    Returns:
        Value at the nested key or default
    """
    value = dictionary
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value


def format_number(number: Union[int, float], decimal_places: int = 0) -> str:
    """
    Format number with thousand separators (Persian style)
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
    
    Returns:
        Formatted number string
    """
    if pd.isna(number):
        return ""
    
    if decimal_places > 0:
        formatted = f"{number:,.{decimal_places}f}"
    else:
        formatted = f"{int(number):,}"
    
    return formatted