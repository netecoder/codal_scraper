"""
Data processing and export functionality for Codal data
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

from .utils import (
    dataframe_columns_to_snake_case,
    parse_codal_response,
    normalize_persian_text
)
from .constants import LETTER_CODES

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Process and export Codal data to various formats
    """
    
    def __init__(self, data: Union[List[Dict], pd.DataFrame] = None):
        """
        Initialize data processor
        
        Args:
            data: Input data (list of dictionaries or DataFrame)
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, list):
            self.df = pd.DataFrame(data) if data else pd.DataFrame()
        else:
            self.df = pd.DataFrame()
        
        # Normalize column names if data exists
        if not self.df.empty:
            self.df = dataframe_columns_to_snake_case(self.df)
            self._process_dates()
            self._normalize_text_fields()
    
    def _process_dates(self) -> None:
        """Process date columns"""
        # Common date column patterns in Codal API
        date_patterns = ['date', 'Date', 'تاریخ', 'DateTime', 'PublishDateTime']
        date_columns = []
        
        for col in self.df.columns:
            for pattern in date_patterns:
                if pattern in col or pattern.lower() in col.lower():
                    date_columns.append(col)
                    break
        
        for col in date_columns:
            try:
                # Try to parse with specific format first (for Persian dates)
                if self.df[col].dtype == 'object' and len(self.df[col]) > 0:
                    # Check if it's a Persian date format (YYYY/MM/DD)
                    sample = str(self.df[col].iloc[0]) if pd.notna(self.df[col].iloc[0]) else ""
                    if '/' in sample and len(sample.split('/')[0]) == 4:
                        # Persian date format - keep as string for now
                        logger.info(f"Column {col} appears to be Persian date, keeping as string")
                        continue
                    
                # Try standard datetime parsing for other formats
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce', format=None)
            except Exception as e:
                logger.warning(f"Could not convert column {col} to datetime: {e}")
    
    def _normalize_text_fields(self) -> None:
        """Normalize Persian text fields"""
        text_columns = ['symbol', 'company_name', 'title', 'subject']
        
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(lambda x: normalize_persian_text(str(x)) if pd.notna(x) else '')
    
    def add_letter_descriptions(self) -> 'DataProcessor':
        """Add human-readable descriptions for letter codes"""
        if 'letter_code' in self.df.columns:
            self.df['letter_description'] = self.df['letter_code'].map(LETTER_CODES)
        return self
    
    def filter_by_letter_code(self, codes: Union[str, List[str]]) -> 'DataProcessor':
        """
        Filter data by letter code(s)
        
        Args:
            codes: Single code or list of codes to filter by
        
        Returns:
            Self for method chaining
        """
        if isinstance(codes, str):
            codes = [codes]
        
        if 'letter_code' in self.df.columns:
            self.df = self.df[self.df['letter_code'].isin(codes)]
        
        return self
    
    def filter_by_date_range(self, 
                            start_date: str = None, 
                            end_date: str = None,
                            date_column: str = 'publish_date') -> 'DataProcessor':
        """
        Filter data by date range
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            date_column: Name of date column to filter
        
        Returns:
            Self for method chaining
        """
        if date_column not in self.df.columns:
            logger.warning(f"Date column '{date_column}' not found")
            return self
        
        if start_date:
            self.df = self.df[self.df[date_column] >= pd.to_datetime(start_date)]
        
        if end_date:
            self.df = self.df[self.df[date_column] <= pd.to_datetime(end_date)]
        
        return self
    
    def filter_by_symbols(self, symbols: Union[str, List[str]]) -> 'DataProcessor':
        """
        Filter data by symbol(s)
        
        Args:
            symbols: Single symbol or list of symbols
        
        Returns:
            Self for method chaining
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if 'symbol' in self.df.columns:
            self.df = self.df[self.df['symbol'].isin(symbols)]
        
        return self
    
    def remove_duplicates(self, 
                         subset: List[str] = None, 
                         keep: str = 'last') -> 'DataProcessor':
        """
        Remove duplicate rows
        
        Args:
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep ('first', 'last', False)
        
        Returns:
            Self for method chaining
        """
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self
    
    def sort_by(self, 
               columns: Union[str, List[str]], 
               ascending: Union[bool, List[bool]] = True) -> 'DataProcessor':
        """
        Sort data by column(s)
        
        Args:
            columns: Column(s) to sort by
            ascending: Sort order
        
        Returns:
            Self for method chaining
        """
        # Check if columns exist
        if isinstance(columns, str):
            if columns not in self.df.columns:
                logger.warning(f"Column '{columns}' not found for sorting. Available columns: {self.df.columns.tolist()[:10]}...")
                return self
            columns_list = [columns]
        else:
            columns_list = columns
            missing = [c for c in columns_list if c not in self.df.columns]
            if missing:
                logger.warning(f"Columns {missing} not found for sorting")
                columns_list = [c for c in columns_list if c in self.df.columns]
                if not columns_list:
                    return self
        
        try:
            self.df = self.df.sort_values(by=columns, ascending=ascending)
        except Exception as e:
            logger.warning(f"Failed to sort by {columns}: {e}")
        
        return self
    
    def aggregate_by_symbol_year(self) -> pd.DataFrame:
        """
        Aggregate data by symbol and year
        
        Returns:
            Aggregated DataFrame
        """
        if 'symbol' not in self.df.columns or 'publish_date' not in self.df.columns:
            logger.warning("Required columns for aggregation not found")
            return self.df
        
        # Extract year from date
        self.df['year'] = pd.to_datetime(self.df['publish_date']).dt.year
        
        # Group and aggregate
        agg_dict = {
            'letter_code': 'count',  # Count of announcements
            'company_name': 'first'  # Keep first company name
        }
        
        # Add more aggregations based on available columns
        if 'audited' in self.df.columns:
            agg_dict['audited'] = lambda x: (x == True).sum()
        
        aggregated = self.df.groupby(['symbol', 'year']).agg(agg_dict)
        aggregated.columns = ['announcement_count', 'company_name'] + list(aggregated.columns[2:])
        
        return aggregated.reset_index()
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for the data
        
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Add column-specific stats
        if 'symbol' in self.df.columns:
            stats['unique_symbols'] = self.df['symbol'].nunique()
            stats['top_symbols'] = self.df['symbol'].value_counts().head(10).to_dict()
        
        if 'letter_code' in self.df.columns:
            stats['unique_letter_codes'] = self.df['letter_code'].nunique()
            stats['letter_code_distribution'] = self.df['letter_code'].value_counts().to_dict()
        
        # Date range
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        if date_cols:
            first_date_col = date_cols[0]
            stats['date_range'] = {
                'min': str(self.df[first_date_col].min()),
                'max': str(self.df[first_date_col].max())
            }
        
        return stats
    
    def to_excel(self, 
                filepath: Union[str, Path], 
                sheet_name: str = 'data',
                index: bool = False) -> None:
        """
        Export data to Excel file
        
        Args:
            filepath: Output file path
            sheet_name: Excel sheet name
            index: Whether to include index
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self.df.to_excel(writer, sheet_name=sheet_name, index=index)
                
                # Add summary sheet if data is large
                if len(self.df) > 100:
                    summary = pd.DataFrame([self.get_summary_stats()])
                    summary.to_excel(writer, sheet_name='summary', index=False)
            
            logger.info(f"Data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export to Excel: {e}")
    
    def to_csv(self, 
              filepath: Union[str, Path], 
              index: bool = False,
              encoding: str = 'utf-8') -> None:
        """
        Export data to CSV file
        
        Args:
            filepath: Output file path
            index: Whether to include index
            encoding: File encoding
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.df.to_csv(filepath, index=index, encoding=encoding)
            logger.info(f"Data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
    
    def to_json(self, 
               filepath: Union[str, Path], 
               orient: str = 'records',
               indent: int = 2) -> None:
        """
        Export data to JSON file
        
        Args:
            filepath: Output file path
            orient: JSON orientation ('records', 'index', 'columns', etc.)
            indent: JSON indentation
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert datetime columns to string for JSON serialization
            df_copy = self.df.copy()
            for col in df_copy.select_dtypes(include=['datetime']).columns:
                df_copy[col] = df_copy[col].astype(str)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(
                    json.loads(df_copy.to_json(orient=orient, force_ascii=False)),
                    f,
                    ensure_ascii=False,
                    indent=indent
                )
            
            logger.info(f"Data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
    
    def to_parquet(self, filepath: Union[str, Path]) -> None:
        """
        Export data to Parquet file
        
        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.df.to_parquet(filepath, index=False)
            logger.info(f"Data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export to Parquet: {e}")
    
    @classmethod
    def from_json_file(cls, filepath: Union[str, Path]) -> 'DataProcessor':
        """
        Load data from JSON file
        
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
                elif 'pages' in data:
                    # Handle paginated data
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
    def from_excel_file(cls, 
                       filepath: Union[str, Path], 
                       sheet_name: Union[str, int] = 0) -> 'DataProcessor':
        """
        Load data from Excel file
        
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