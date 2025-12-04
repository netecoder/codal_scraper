"""
Utility functions for data processing and conversion

This module provides helper functions for text normalization, date conversion,
data cleaning, and other common operations.
"""

import re
import logging
from typing import Any, Dict, Iterator, List, Optional, Sized, Union

import pandas as pd
from jdatetime import date as jd
from jdatetime import datetime as jdt

from .constants import FA_TO_EN_DIGITS, AR_TO_EN_DIGITS, AR_TO_FA_LETTER


logger = logging.getLogger(__name__)


# ============== Dictionary Utilities ==============

def clean_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove None and -1 values from dictionary.
    
    Args:
        dictionary: Input dictionary to clean
        
    Returns:
        New dictionary without None or -1 values
        
    Example:
        >>> clean_dict({'a': 1, 'b': -1, 'c': None})
        {'a': 1}
    """
    return {k: v for k, v in dictionary.items() if v not in [None, -1, "-1"]}


def safe_get(dictionary: Dict, *keys, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        dictionary: The dictionary to search
        *keys: Keys to traverse
        default: Default value if key not found
    
    Returns:
        Value at the nested key or default
        
    Example:
        >>> d = {'a': {'b': {'c': 1}}}
        >>> safe_get(d, 'a', 'b', 'c')
        1
        >>> safe_get(d, 'a', 'x', default='not found')
        'not found'
    """
    value = dictionary
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries, later ones override earlier.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


# ============== Text Normalization ==============

def normalize_persian_text(text: str) -> str:
    """
    Normalize Persian/Arabic characters for consistency.
    
    This function:
    - Converts Arabic characters to Persian equivalents
    - Removes zero-width characters
    - Normalizes whitespace
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
        
    Example:
        >>> normalize_persian_text("كتاب يوسف")
        'کتاب یوسف'
    """
    if not text:
        return ''
    
    text = str(text)
    
    # Convert Arabic to Persian letters
    for ar, fa in AR_TO_FA_LETTER.items():
        text = text.replace(ar, fa)
    
    # Remove zero-width characters
    zero_width_chars = ['\u200c', '\u200f', '\u200e', '\u200d', '\u200b', '\ufeff']
    for char in zero_width_chars:
        text = text.replace(char, '')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def persian_to_english_digits(text: str) -> str:
    """
    Convert Persian/Arabic digits to English (ASCII) digits.
    
    Args:
        text: Input text with Persian/Arabic digits
        
    Returns:
        Text with English digits
        
    Example:
        >>> persian_to_english_digits("۱۳۹۸/۰۵/۱۵")
        '1398/05/15'
    """
    if not text:
        return text
    
    text = str(text)
    
    # Convert Persian digits
    for fa, en in FA_TO_EN_DIGITS.items():
        text = text.replace(fa, en)
    
    # Convert Arabic digits
    for ar, en in AR_TO_EN_DIGITS.items():
        text = text.replace(ar, en)
    
    return text


def english_to_persian_digits(text: str) -> str:
    """
    Convert English digits to Persian digits.
    
    Args:
        text: Input text with English digits
        
    Returns:
        Text with Persian digits
        
    Example:
        >>> english_to_persian_digits("1398/05/15")
        '۱۳۹۸/۰۵/۱۵'
    """
    if not text:
        return text
    
    text = str(text)
    en_to_fa = {v: k for k, v in FA_TO_EN_DIGITS.items()}
    
    for en, fa in en_to_fa.items():
        text = text.replace(en, fa)
    
    return text


def clean_symbol(symbol: str) -> str:
    """
    Clean and normalize stock symbol.
    
    Args:
        symbol: Stock symbol to clean
        
    Returns:
        Cleaned symbol
        
    Example:
        >>> clean_symbol("  فولاد۱  ")
        'فولاد1'
    """
    if not symbol:
        return ""
    
    # Strip whitespace
    symbol = symbol.strip()
    
    # Convert Persian digits to English
    symbol = persian_to_english_digits(symbol)
    
    # Normalize Persian text
    symbol = normalize_persian_text(symbol)
    
    # Remove special characters (keep alphanumeric and Persian letters)
    symbol = re.sub(r'[^\w\u0600-\u06FF]+', '', symbol)
    
    # Uppercase only English letters (Persian doesn't have case)
    return ''.join(
        c.upper() if c.isascii() and c.isalpha() else c
        for c in symbol
    )


# ============== Date Utilities ==============

def datetime_to_num(dt: Union[str, None]) -> Optional[int]:
    """
    Convert datetime string to numeric format (YYYYMMDDHHmmss).
    
    Args:
        dt: Datetime string to convert
        
    Returns:
        Integer representation or None if conversion fails
        
    Example:
        >>> datetime_to_num("1402/05/15 10:30:00")
        14020515103000
    """
    if not dt or dt == "":
        return None
    
    try:
        # Remove non-numeric characters
        dt_clean = re.sub(r'[^0-9]', '', str(dt))
        
        if not dt_clean:
            return None
        
        # Pad with zeros to get 14 digits (YYYYMMDDHHmmss)
        dt_len = len(dt_clean)
        if dt_len > 14:
            dt_clean = dt_clean[:14]
        elif dt_len < 14:
            dt_clean = dt_clean.ljust(14, '0')
        
        return int(dt_clean)
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Error converting datetime '{dt}': {e}")
        return None


def num_to_datetime(
    num: Union[int, str],
    include_time: bool = True,
    date_sep: str = "/",
    time_sep: str = ":",
    dt_sep: str = " "
) -> str:
    """
    Convert numeric datetime to string format.
    
    Args:
        num: Numeric datetime (YYYYMMDDHHmmss)
        include_time: If True, include time component
        date_sep: Separator for date components
        time_sep: Separator for time components
        dt_sep: Separator between date and time
    
    Returns:
        Formatted datetime string
        
    Example:
        >>> num_to_datetime(14020515103000)
        '1402/05/15 10:30:00'
    """
    num_str = str(num).zfill(14)
    
    date_part = f"{num_str[0:4]}{date_sep}{num_str[4:6]}{date_sep}{num_str[6:8]}"
    
    if include_time:
        time_part = f"{num_str[8:10]}{time_sep}{num_str[10:12]}{time_sep}{num_str[12:14]}"
        return f"{date_part}{dt_sep}{time_part}"
    
    return date_part


def gregorian_to_shamsi(date: Union[str, int]) -> str:
    """
    Convert Gregorian date (YYYYMMDD) to Shamsi (Persian) calendar.
    
    Args:
        date: Gregorian date as YYYYMMDD string or integer
        
    Returns:
        Persian date as YYYY/MM/DD string
        
    Example:
        >>> gregorian_to_shamsi("20230815")
        '1402/05/24'
    """
    date_str = str(date)
    
    if len(date_str) != 8:
        raise ValueError(f"Date must be in YYYYMMDD format, got: {date_str}")
    
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    
    shamsi_date = jd.fromgregorian(day=day, month=month, year=year)
    return shamsi_date.strftime("%Y/%m/%d")


def shamsi_to_gregorian(date: str) -> str:
    """
    Convert Shamsi (Persian) date to Gregorian (YYYYMMDD).
    
    Args:
        date: Persian date as YYYY/MM/DD or YYYYMMDD string
        
    Returns:
        Gregorian date as YYYYMMDD string
        
    Example:
        >>> shamsi_to_gregorian("1402/05/24")
        '20230815'
    """
    # Remove separators
    date_clean = date.replace('/', '')
    
    # Parse the Shamsi date
    date_obj = jdt.strptime(date_clean, "%Y%m%d")
    
    # Convert to Gregorian
    greg_date = date_obj.togregorian()
    
    return greg_date.strftime("%Y%m%d")


def calculate_date_range(year: int) -> tuple:
    """
    Calculate the start and end dates for a Persian calendar year.
    
    Args:
        year: Persian calendar year
    
    Returns:
        Tuple of (start_date, end_date) in YYYY/MM/DD format
        
    Example:
        >>> calculate_date_range(1402)
        ('1402/01/01', '1402/12/29')
    """
    start_date = f"{year}/01/01"
    end_date = f"{year}/12/29"
    
    return start_date, end_date


def is_valid_persian_date(date_str: str) -> bool:
    """
    Check if a string is a valid Persian date.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        from .validators import InputValidator
        InputValidator(date_str).is_date()
        return True
    except Exception:
        return False


# ============== Value Conversion ==============

def value_to_float(value: Union[str, int, float]) -> float:
    """
    Convert formatted values to float.
    Handles K (thousands), M (millions), B (billions) suffixes.
    
    Args:
        value: Value to convert
        
    Returns:
        Float value
        
    Example:
        >>> value_to_float("1.5M")
        1500000.0
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return 0.0
    
    # Remove commas and whitespace
    value = value.replace(',', '').replace(' ', '').strip()
    
    if not value:
        return 0.0
    
    # Handle suffixes
    multipliers = {
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
        'T': 1_000_000_000_000,
        'k': 1_000,
        'm': 1_000_000,
        'b': 1_000_000_000,
        't': 1_000_000_000_000,
    }
    
    for suffix, multiplier in multipliers.items():
        if value.endswith(suffix):
            try:
                number = float(value[:-1].strip())
                return number * multiplier
            except ValueError:
                return 0.0
    
    try:
        return float(value)
    except ValueError:
        return 0.0


def format_number(number: Union[int, float], decimal_places: int = 0) -> str:
    """
    Format number with thousand separators (Persian style).
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
    
    Returns:
        Formatted number string
        
    Example:
        >>> format_number(1234567)
        '1,234,567'
    """
    if pd.isna(number):
        return ""
    
    try:
        if decimal_places > 0:
            return f"{float(number):,.{decimal_places}f}"
        else:
            return f"{int(number):,}"
    except (ValueError, TypeError):
        return str(number)


# ============== String Utilities ==============

def to_snake_case(name: str) -> str:
    """
    Convert CamelCase or PascalCase to snake_case.
    
    Args:
        name: String to convert
        
    Returns:
        snake_case string
        
    Example:
        >>> to_snake_case("PublishDateTime")
        'publish_date_time'
    """
    # Insert underscore before uppercase letters
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    
    return name.lower()


def to_camel_case(name: str) -> str:
    """
    Convert snake_case to camelCase.
    
    Args:
        name: String to convert
        
    Returns:
        camelCase string
        
    Example:
        >>> to_camel_case("publish_date_time")
        'publishDateTime'
    """
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


# ============== DataFrame Utilities ==============

def dataframe_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all DataFrame column names to snake_case.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with snake_case column names
    """
    df = df.copy()
    df.columns = [to_snake_case(col) for col in df.columns]
    return df


def parse_codal_response(response: Dict) -> pd.DataFrame:
    """
    Parse Codal API response into a DataFrame.
    
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
    
    return df


# ============== Collection Utilities ==============

def chunk_list(lst: List, chunk_size: int) -> Iterator[List]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of the list
        
    Example:
        >>> list(chunk_list([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def flatten_list(nested_list: List[List]) -> List:
    """
    Flatten a nested list.
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def unique_preserve_order(lst: List) -> List:
    """
    Remove duplicates from list while preserving order.
    
    Args:
        lst: List with potential duplicates
        
    Returns:
        List with duplicates removed
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ============== Progress Utilities ==============

def progress_iterator(
    items: Sized,
    desc: str = "Processing",
    disable: bool = False,
    **kwargs
) -> Iterator:
    """
    Wrap iterable with progress bar if tqdm is available.
    
    Args:
        items: Iterable to wrap
        desc: Progress bar description
        disable: If True, disable progress bar
        **kwargs: Additional arguments for tqdm
        
    Returns:
        Iterator (with or without progress bar)
    """
    try:
        from tqdm import tqdm
        return tqdm(items, desc=desc, disable=disable, **kwargs)
    except ImportError:
        return iter(items)


# ============== URL Utilities ==============

def build_full_url(path: str, base_url: str = "https://codal.ir") -> str:
    """
    Build full URL from a path.
    
    Args:
        path: URL path (may be relative or absolute)
        base_url: Base URL to prepend if path is relative
        
    Returns:
        Full URL
    """
    if not path:
        return ""
    
    if path.startswith('http://') or path.startswith('https://'):
        return path
    
    if path.startswith('/'):
        return f"{base_url}{path}"
    
    return f"{base_url}/{path}"


def extract_tracing_no_from_url(url: str) -> Optional[str]:
    """
    Extract tracing number from Codal URL.
    
    Args:
        url: Codal report URL
        
    Returns:
        Tracing number or None
    """
    if not url:
        return None
    
    patterns = [
        r'LetterSerial=(\d+)',
        r'TracingNo=(\d+)',
        r'/(\d+)$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None