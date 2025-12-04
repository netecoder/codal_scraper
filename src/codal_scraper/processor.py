"""
Data processing and export functionality for Codal data

This module provides the DataProcessor class for cleaning, transforming,
filtering, and exporting Codal data in various formats.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .utils import (
    dataframe_columns_to_snake_case,
    normalize_persian_text,
    to_snake_case
)
from .constants import LETTER_CODES, LETTER_DESCRIPTIONS_TO_CODES
from .exceptions import CodalScraperError


logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Process and export Codal data to various formats.
    
    This class provides a fluent interface for data transformation,
    filtering, and export operations on Codal announcement data.
    
    Features:
    - Fluent API with method chaining
    - Automatic column name normalization
    - Date and text field processing
    - Multiple export formats (CSV, Excel, JSON, Parquet)
    - Summary statistics generation
    
    Example:
        >>> processor = DataProcessor(letters)
        >>> (processor
        ...     .filter_by_letter_code("ن-45")
        ...     .filter_by_date_range("1402/01/01", "1402/06/31")
        ...     .sort_by("publish_date_time", ascending=False)
        ...     .to_excel("output.xlsx"))
    
    Attributes:
        df: The underlying pandas DataFrame
    """
    
    def __init__(
        self,
        data: Union[List[Dict], pd.DataFrame, None] = None,
        normalize_columns: bool = True,
        process_dates: bool = True,
        normalize_text: bool = True
    ):
        """
        Initialize data processor.
        
        Args:
            data: Input data (list of dictionaries or DataFrame)
            normalize_columns: Convert column names to snake_case
            process_dates: Process date columns
            normalize_text: Normalize Persian text in text fields
        """
        # Convert input to DataFrame
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        elif isinstance(data, list) and data:
            self.df = pd.DataFrame(data)
        else:
            self.df = pd.DataFrame()
        
        # Apply initial processing
        if not self.df.empty:
            if normalize_columns:
                self.df = dataframe_columns_to_snake_case(self.df)
            if process_dates:
                self._process_dates()
            if normalize_text:
                self._normalize_text_fields()
        
        # Track transformations for logging
        self._transformations: List[str] = []
    
    def __len__(self) -> int:
        """Return number of rows"""
        return len(self.df)
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"DataProcessor(rows={len(self.df)}, "
            f"columns={len(self.df.columns)}, "
            f"transformations={len(self._transformations)})"
        )
    
    # ============== Initial Processing ==============
    
    def _process_dates(self) -> None:
        """Process date columns"""
        date_patterns = ['date', 'Date', 'تاریخ', 'DateTime', 'time', 'Time']
        
        for col in self.df.columns:
            is_date_col = any(
                pattern.lower() in col.lower()
                for pattern in date_patterns
            )
            
            if not is_date_col:
                continue
            
            try:
                if self.df[col].dtype == 'object' and len(self.df) > 0:
                    sample = str(self.df[col].iloc[0]) if pd.notna(self.df[col].iloc[0]) else ""
                    
                    # Check for Persian date format (YYYY/MM/DD)
                    if '/' in sample and len(sample.split('/')[0]) == 4:
                        logger.debug(f"Column {col} is Persian date format")
                        continue
                    
                # Try standard datetime parsing
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                
            except Exception as e:
                logger.debug(f"Could not convert {col} to datetime: {e}")
    
    def _normalize_text_fields(self) -> None:
        """Normalize Persian text in common text fields"""
        text_columns = [
            'symbol', 'company_name', 'title', 'subject',
            'company', 'name', 'description'
        ]
        
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(
                    lambda x: normalize_persian_text(str(x)) if pd.notna(x) else ''
                )
    
    # ============== Fluent Transformations ==============
    
    def add_letter_descriptions(self) -> 'DataProcessor':
        """
        Add human-readable descriptions for letter codes.
        
        Returns:
            Self for method chaining
        """
        if 'letter_code' in self.df.columns:
            self.df['letter_description'] = self.df['letter_code'].map(LETTER_CODES)
            self._transformations.append('add_letter_descriptions')
        else:
            logger.warning("Column 'letter_code' not found")
        
        return self
    
    def filter_by_letter_code(
        self,
        codes: Union[str, List[str]]
    ) -> 'DataProcessor':
        """
        Filter data by letter code(s).
        
        Args:
            codes: Single code or list of codes to filter by
        
        Returns:
            Self for method chaining
            
        Example:
            >>> processor.filter_by_letter_code(["ن-45", "ن-10"])
        """
        if isinstance(codes, str):
            codes = [codes]
        
        if 'letter_code' in self.df.columns:
            original_count = len(self.df)
            self.df = self.df[self.df['letter_code'].isin(codes)]
            self._transformations.append(
                f'filter_by_letter_code({codes}): {original_count} -> {len(self.df)}'
            )
        else:
            logger.warning("Column 'letter_code' not found")
        
        return self
    
    def filter_by_date_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_column: str = 'publish_date_time'
    ) -> 'DataProcessor':
        """
        Filter data by date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            date_column: Name of date column to filter
        
        Returns:
            Self for method chaining
        """
        # Try alternative column names
        possible_columns = [
            date_column,
            'publish_date',
            'sent_date_time',
            'sent_date',
            'date'
        ]
        
        actual_column = None
        for col in possible_columns:
            if col in self.df.columns:
                actual_column = col
                break
        
        if not actual_column:
            logger.warning(f"No date column found. Tried: {possible_columns}")
            return self
        
        original_count = len(self.df)
        
        try:
            # Convert column to string for comparison if it's a Persian date
            col_data = self.df[actual_column].astype(str)
            
            if start_date:
                self.df = self.df[col_data >= start_date]
            
            if end_date:
                col_data = self.df[actual_column].astype(str)
                self.df = self.df[col_data <= end_date]
            
            self._transformations.append(
                f'filter_by_date_range({start_date} to {end_date}): '
                f'{original_count} -> {len(self.df)}'
            )
            
        except Exception as e:
            logger.warning(f"Failed to filter by date range: {e}")
        
        return self
    
    def filter_by_symbols(
        self,
        symbols: Union[str, List[str]]
    ) -> 'DataProcessor':
        """
        Filter data by symbol(s).
        
        Args:
            symbols: Single symbol or list of symbols
        
        Returns:
            Self for method chaining
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Normalize symbols for comparison
        symbols = [normalize_persian_text(s) for s in symbols]
        
        if 'symbol' in self.df.columns:
            original_count = len(self.df)
            
            # Normalize the column for comparison
            normalized_col = self.df['symbol'].apply(
                lambda x: normalize_persian_text(str(x)) if pd.notna(x) else ''
            )
            self.df = self.df[normalized_col.isin(symbols)]
            
            self._transformations.append(
                f'filter_by_symbols({symbols}): {original_count} -> {len(self.df)}'
            )
        else:
            logger.warning("Column 'symbol' not found")
        
        return self
    
    def filter_by_condition(
        self,
        condition: Callable[[pd.DataFrame], pd.Series],
        description: str = "custom"
    ) -> 'DataProcessor':
        """
        Filter data by a custom condition.
        
        Args:
            condition: Function that takes DataFrame and returns boolean Series
            description: Description of the filter for logging
        
        Returns:
            Self for method chaining
            
        Example:
            >>> processor.filter_by_condition(
            ...     lambda df: df['has_excel'] == True,
            ...     description="has_excel"
            ... )
        """
        original_count = len(self.df)
        
        try:
            mask = condition(self.df)
            self.df = self.df[mask]
            self._transformations.append(
                f'filter_by_condition({description}): {original_count} -> {len(self.df)}'
            )
        except Exception as e:
            logger.warning(f"Failed to apply condition filter: {e}")
        
        return self
    
    def remove_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: str = 'last'
    ) -> 'DataProcessor':
        """
        Remove duplicate rows.
        
        Args:
            subset: Columns to consider for duplicates (None = all columns)
            keep: Which duplicates to keep ('first', 'last', False)
        
        Returns:
            Self for method chaining
        """
        original_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        
        removed = original_count - len(self.df)
        self._transformations.append(
            f'remove_duplicates(subset={subset}, keep={keep}): removed {removed}'
        )
        
        return self
    
    def sort_by(
        self,
        columns: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True
    ) -> 'DataProcessor':
        """
        Sort data by column(s).
        
        Args:
            columns: Column(s) to sort by
            ascending: Sort order
        
        Returns:
            Self for method chaining
        """
        if isinstance(columns, str):
            columns = [columns]
        
        # Filter to existing columns
        existing = [c for c in columns if c in self.df.columns]
        missing = [c for c in columns if c not in self.df.columns]
        
        if missing:
            logger.warning(f"Columns not found for sorting: {missing}")
        
        if existing:
            self.df = self.df.sort_values(by=existing, ascending=ascending)
            self._transformations.append(f'sort_by({existing}, ascending={ascending})')
        
        return self
    
    def select_columns(
        self,
        columns: List[str]
    ) -> 'DataProcessor':
        """
        Select specific columns.
        
        Args:
            columns: List of column names to keep
        
        Returns:
            Self for method chaining
        """
        existing = [c for c in columns if c in self.df.columns]
        missing = [c for c in columns if c not in self.df.columns]
        
        if missing:
            logger.warning(f"Columns not found: {missing}")
        
        if existing:
            self.df = self.df[existing]
            self._transformations.append(f'select_columns({existing})')
        
        return self
    
    def drop_columns(
        self,
        columns: List[str]
    ) -> 'DataProcessor':
        """
        Drop specific columns.
        
        Args:
            columns: List of column names to drop
        
        Returns:
            Self for method chaining
        """
        existing = [c for c in columns if c in self.df.columns]
        
        if existing:
            self.df = self.df.drop(columns=existing)
            self._transformations.append(f'drop_columns({existing})')
        
        return self
    
    def rename_columns(
        self,
        mapping: Dict[str, str]
    ) -> 'DataProcessor':
        """
        Rename columns.
        
        Args:
            mapping: Dictionary of old_name -> new_name
        
        Returns:
            Self for method chaining
        """
        # Filter to existing columns
        valid_mapping = {k: v for k, v in mapping.items() if k in self.df.columns}
        
        if valid_mapping:
            self.df = self.df.rename(columns=valid_mapping)
            self._transformations.append(f'rename_columns({valid_mapping})')
        
        return self
    
    def add_column(
        self,
        name: str,
        value: Union[Any, Callable[[pd.DataFrame], pd.Series]]
    ) -> 'DataProcessor':
        """
        Add a new column.
        
        Args:
            name: Column name
            value: Static value or function that takes DataFrame and returns Series
        
        Returns:
            Self for method chaining
        """
        try:
            if callable(value):
                self.df[name] = value(self.df)
            else:
                self.df[name] = value
            
            self._transformations.append(f'add_column({name})')
            
        except Exception as e:
            logger.warning(f"Failed to add column '{name}': {e}")
        
        return self
    
    def apply_to_column(
        self,
        column: str,
        func: Callable,
        new_column: Optional[str] = None
    ) -> 'DataProcessor':
        """
        Apply a function to a column.
        
        Args:
            column: Column to apply function to
            func: Function to apply
            new_column: Name for result (if None, overwrites original)
        
        Returns:
            Self for method chaining
        """
        if column not in self.df.columns:
            logger.warning(f"Column '{column}' not found")
            return self
        
        target = new_column or column
        
        try:
            self.df[target] = self.df[column].apply(func)
            self._transformations.append(f'apply_to_column({column} -> {target})')
        except Exception as e:
            logger.warning(f"Failed to apply function to column: {e}")
        
        return self
    
    def fill_na(
        self,
        value: Any = '',
        columns: Optional[List[str]] = None
    ) -> 'DataProcessor':
        """
        Fill NA/NaN values.
        
        Args:
            value: Value to fill with
            columns: Specific columns to fill (None = all columns)
        
        Returns:
            Self for method chaining
        """
        if columns:
            existing = [c for c in columns if c in self.df.columns]
            self.df[existing] = self.df[existing].fillna(value)
        else:
            self.df = self.df.fillna(value)
        
        self._transformations.append(f'fill_na({value})')
        return self
    
    def reset_index(self) -> 'DataProcessor':
        """
        Reset DataFrame index.
        
        Returns:
            Self for method chaining
        """
        self.df = self.df.reset_index(drop=True)
        self._transformations.append('reset_index')
        return self
    
    # ============== Aggregation Methods ==============
    
    def aggregate_by_symbol_year(self) -> pd.DataFrame:
        """
        Aggregate data by symbol and year.
        
        Returns:
            Aggregated DataFrame
        """
        if 'symbol' not in self.df.columns:
            logger.warning("Column 'symbol' not found")
            return self.df
        
        # Find date column
        date_col = None
        for col in ['publish_date_time', 'publish_date', 'sent_date_time', 'date']:
            if col in self.df.columns:
                date_col = col
                break
        
        if not date_col:
            logger.warning("No date column found for aggregation")
            return self.df
        
        # Extract year
        df_copy = self.df.copy()
        
        try:
            # Try to extract year from string date (Persian format)
            df_copy['year'] = df_copy[date_col].astype(str).str[:4]
        except Exception:
            try:
                df_copy['year'] = pd.to_datetime(df_copy[date_col]).dt.year
            except Exception as e:
                logger.warning(f"Failed to extract year: {e}")
                return self.df
        
        # Aggregate
        agg_dict = {'year': 'count'}
        
        if 'company_name' in df_copy.columns:
            agg_dict['company_name'] = 'first'
        if 'letter_code' in df_copy.columns:
            agg_dict['letter_code'] = lambda x: list(x.unique())
        
        aggregated = df_copy.groupby(['symbol', 'year']).agg(agg_dict)
        aggregated = aggregated.rename(columns={'year': 'announcement_count'})
        
        return aggregated.reset_index()
    
    def group_by(
        self,
        columns: Union[str, List[str]],
        aggregations: Dict[str, Union[str, Callable]]
    ) -> pd.DataFrame:
        """
        Group by columns and aggregate.
        
        Args:
            columns: Column(s) to group by
            aggregations: Dict of column -> aggregation function
        
        Returns:
            Aggregated DataFrame
        """
        if isinstance(columns, str):
            columns = [columns]
        
        # Filter to existing columns
        existing = [c for c in columns if c in self.df.columns]
        valid_aggs = {k: v for k, v in aggregations.items() if k in self.df.columns}
        
        if not existing or not valid_aggs:
            logger.warning("Invalid columns for grouping")
            return self.df
        
        return self.df.groupby(existing).agg(valid_aggs).reset_index()
    
    # ============== Statistics ==============
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the data.
        
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'column_count': len(self.df.columns),
            'memory_usage_mb': round(
                self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2
            ),
            'transformations_applied': self._transformations.copy()
        }
        
        # Symbol statistics
        if 'symbol' in self.df.columns:
            stats['unique_symbols'] = self.df['symbol'].nunique()
            stats['top_symbols'] = self.df['symbol'].value_counts().head(10).to_dict()
        
        # Letter code statistics
        if 'letter_code' in self.df.columns:
            stats['unique_letter_codes'] = self.df['letter_code'].nunique()
            stats['letter_code_distribution'] = self.df['letter_code'].value_counts().to_dict()
        
        # Date range
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        if date_cols:
            first_date_col = date_cols[0]
            try:
                stats['date_range'] = {
                    'column': first_date_col,
                    'min': str(self.df[first_date_col].min()),
                    'max': str(self.df[first_date_col].max())
                }
            except Exception:
                pass
        
        # NA counts
        na_counts = self.df.isna().sum()
        stats['columns_with_na'] = {
            col: int(count) for col, count in na_counts.items() if count > 0
        }
        
        return stats
    
    def get_column_info(self) -> pd.DataFrame:
        info = []
        for col in self.df.columns:
            series = self.df[col]
            try:
                unique_count = series.nunique()
            except TypeError:
                # Fallback for unhashable entries (e.g., dicts/lists)
                unique_count = series.astype(str).nunique()

            info.append({
                'column': col,
                'dtype': str(series.dtype),
                'non_null_count': series.notna().sum(),
                'null_count': series.isna().sum(),
                'unique_count': unique_count,
                'sample_value': str(series.iloc[0]) if len(series) > 0 else ''
            })
        return pd.DataFrame(info)

    
    def describe(self) -> pd.DataFrame:
        """
        Get descriptive statistics.
        
        Returns:
            DataFrame with descriptive statistics
        """
        return self.df.describe(include='all')
    
    # ============== Export Methods ==============
    
    def to_excel(
        self,
        filepath: Union[str, Path],
        sheet_name: str = 'data',
        index: bool = False,
        include_summary: bool = True
    ) -> 'DataProcessor':
        """
        Export data to Excel file.
        
        Args:
            filepath: Output file path
            sheet_name: Excel sheet name
            index: Whether to include index
            include_summary: Whether to add summary sheet
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self.df.to_excel(writer, sheet_name=sheet_name, index=index)
                
                if include_summary and len(self.df) > 0:
                    summary_df = pd.DataFrame([self.get_summary_stats()])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    col_info = self.get_column_info()
                    col_info.to_excel(writer, sheet_name='Column Info', index=False)
            
            logger.info(f"Data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export to Excel: {e}")
            raise
        
        return self
    
    def to_csv(
        self,
        filepath: Union[str, Path],
        index: bool = False,
        encoding: str = 'utf-8-sig'
    ) -> 'DataProcessor':
        """
        Export data to CSV file.
        
        Args:
            filepath: Output file path
            index: Whether to include index
            encoding: File encoding (utf-8-sig for Excel compatibility)
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.df.to_csv(filepath, index=index, encoding=encoding)
            logger.info(f"Data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise
        
        return self
    
    def to_json(
        self,
        filepath: Union[str, Path],
        orient: str = 'records',
        indent: int = 2
    ) -> 'DataProcessor':
        """
        Export data to JSON file.
        
        Args:
            filepath: Output file path
            orient: JSON orientation ('records', 'index', 'columns', etc.)
            indent: JSON indentation
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert datetime columns to string
            df_copy = self.df.copy()
            for col in df_copy.select_dtypes(include=['datetime64']).columns:
                df_copy[col] = df_copy[col].astype(str)
            
            # Export
            with open(filepath, 'w', encoding='utf-8') as f:
                data = json.loads(df_copy.to_json(orient=orient, force_ascii=False))
                json.dump(data, f, ensure_ascii=False, indent=indent)
            
            logger.info(f"Data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            raise
        
        return self
    
    def to_parquet(
        self,
        filepath: Union[str, Path],
        compression: str = 'snappy'
    ) -> 'DataProcessor':
        """
        Export data to Parquet file.
        
        Args:
            filepath: Output file path
            compression: Compression algorithm
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.df.to_parquet(filepath, index=False, compression=compression)
            logger.info(f"Data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export to Parquet: {e}")
            raise
        
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get the underlying DataFrame.
        
        Returns:
            Copy of the DataFrame
        """
        return self.df.copy()
    
    def to_dict(self, orient: str = 'records') -> Union[List, Dict]:
        """
        Convert to dictionary.
        
        Args:
            orient: Orientation for conversion
            
        Returns:
            Data as dictionary or list of dictionaries
        """
        return self.df.to_dict(orient=orient)
    
    # ============== Class Methods for Loading ==============
    
    @classmethod
    def from_json_file(
        cls,
        filepath: Union[str, Path]
    ) -> 'DataProcessor':
        """
        Load data from JSON file.
        
        Args:
            filepath: JSON file path
        
        Returns:
            DataProcessor instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                if 'Letters' in data:
                    data = data['Letters']
                elif 'data' in data:
                    data = data['data']
                elif 'pages' in data:
                    all_records = []
                    for page_data in data['pages'].values():
                        if page_data:
                            all_records.extend(page_data)
                    data = all_records
            
            return cls(data)
            
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            raise
    
    @classmethod
    def from_excel_file(
        cls,
        filepath: Union[str, Path],
        sheet_name: Union[str, int] = 0
    ) -> 'DataProcessor':
        """
        Load data from Excel file.
        
        Args:
            filepath: Excel file path
            sheet_name: Sheet name or index
        
        Returns:
            DataProcessor instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            return cls(df)
        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            raise
    
    @classmethod
    def from_csv_file(
        cls,
        filepath: Union[str, Path],
        encoding: str = 'utf-8'
    ) -> 'DataProcessor':
        """
        Load data from CSV file.
        
        Args:
            filepath: CSV file path
            encoding: File encoding
        
        Returns:
            DataProcessor instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            return cls(df)
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
    
    @classmethod
    def from_parquet_file(
        cls,
        filepath: Union[str, Path]
    ) -> 'DataProcessor':
        """
        Load data from Parquet file.
        
        Args:
            filepath: Parquet file path
        
        Returns:
            DataProcessor instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            df = pd.read_parquet(filepath)
            return cls(df)
        except Exception as e:
            logger.error(f"Failed to load Parquet file: {e}")
            raise
    
    # ============== Utility Methods ==============
    
    def copy(self) -> 'DataProcessor':
        """
        Create a copy of this processor.
        
        Returns:
            New DataProcessor with copied data
        """
        new_processor = DataProcessor(self.df.copy(), normalize_columns=False)
        new_processor._transformations = self._transformations.copy()
        return new_processor
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Get first n rows.
        
        Args:
            n: Number of rows
            
        Returns:
            DataFrame with first n rows
        """
        return self.df.head(n)
    
    def tail(self, n: int = 5) -> pd.DataFrame:
        """
        Get last n rows.
        
        Args:
            n: Number of rows
            
        Returns:
            DataFrame with last n rows
        """
        return self.df.tail(n)
    
    def sample(self, n: int = 5) -> pd.DataFrame:
        """
        Get random sample of rows.
        
        Args:
            n: Number of rows
            
        Returns:
            DataFrame with random sample
        """
        return self.df.sample(min(n, len(self.df)))
    
    def get_transformations(self) -> List[str]:
        """
        Get list of applied transformations.
        
        Returns:
            List of transformation descriptions
        """
        return self._transformations.copy()
    
    def print_info(self) -> None:
        """Print information about the data"""
        stats = self.get_summary_stats()
        
        print(f"\n{'='*50}")
        print(f"DataProcessor Summary")
        print(f"{'='*50}")
        print(f"Total Records: {stats['total_records']}")
        print(f"Columns: {stats['column_count']}")
        print(f"Memory Usage: {stats['memory_usage_mb']} MB")
        
        if 'unique_symbols' in stats:
            print(f"Unique Symbols: {stats['unique_symbols']}")
        
        if 'date_range' in stats:
            print(f"Date Range: {stats['date_range']['min']} to {stats['date_range']['max']}")
        
        if self._transformations:
            print(f"\nTransformations Applied ({len(self._transformations)}):")
            for t in self._transformations[-5:]:  # Show last 5
                print(f"  - {t}")
        
        print(f"{'='*50}\n")