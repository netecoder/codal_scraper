"""
Example: Fetch Board of Directors changes from Codal
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from codal_scraper import CodalClient, DataProcessor
from codal_scraper.constants import YEAR_RANGES, CURRENT_DATE


def fetch_board_changes_example():
    """
    Example of fetching board of directors changes (ن-45) from Codal
    """
    
    # Initialize client
    client = CodalClient(retry_count=3, timeout=30)
    
    # Search for board changes in the last 3 years
    print("Fetching board of directors changes...")
    
    # Get date range for years 1401-1403
    start_date = YEAR_RANGES[1401][0]  # "1401/01/01"
    end_date = YEAR_RANGES[1403][1]    # "1403/12/29"
    
    # Search for board changes
    board_changes = client.search_board_changes(
        from_date=start_date,
        to_date=end_date,
        company_type='1'  # Only main stock exchange
    )
    
    print(f"Found {len(board_changes)} board change announcements")
    
    # Process the data
    processor = DataProcessor(board_changes)
    
    # Print available columns to debug
    print(f"\nAvailable columns: {processor.df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    # Add letter descriptions
    processor.add_letter_descriptions()
    
    # Find the correct column names
    symbol_col = None
    tracing_col = None
    date_col = None
    
    for col in processor.df.columns:
        col_lower = col.lower()
        if 'symbol' in col_lower:
            symbol_col = col
        elif 'tracing' in col_lower or 'tracingno' in col_lower:
            tracing_col = col
        elif 'publish' in col_lower and 'date' in col_lower:
            date_col = col
        elif 'sent_date' in col_lower or 'sentdate' in col_lower:
            date_col = col  # Alternative date column
    
    # Filter to remove duplicates if columns exist
    if symbol_col and tracing_col:
        processor.remove_duplicates(subset=[symbol_col, tracing_col])
    elif tracing_col:
        processor.remove_duplicates(subset=[tracing_col])
    else:
        processor.remove_duplicates()
    
    # Sort by date if date column exists
    if date_col:
        try:
            processor.sort_by(date_col, ascending=False)
        except KeyError:
            print(f"Warning: Could not sort by {date_col}")
    
    # Get summary statistics
    stats = processor.get_summary_stats()
    print("\nSummary Statistics:")
    print(f"  Total records: {stats.get('total_records', 0)}")
    print(f"  Total unique symbols: {stats.get('unique_symbols', 0)}")
    
    # Safely print date range (handle Unicode issues)
    date_range = stats.get('date_range', {})
    if date_range:
        try:
            print(f"  Date range: {date_range}")
        except UnicodeEncodeError:
            print(f"  Date range: Available (Unicode display issue)")
    else:
        print(f"  Date range: No date range found")
    
    # Export to Excel
    output_path = Path('output') / 'board_changes.xlsx'
    processor.to_excel(output_path)
    print(f"\nData exported to: {output_path}")
    
    # Also export to Parquet for faster loading
    parquet_path = Path('output') / 'board_changes.parquet'
    processor.to_parquet(parquet_path)
    print(f"Data also saved as Parquet: {parquet_path}")
    
    return processor


def fetch_specific_symbol_example(symbol: str = 'فولاد'):
    """
    Example of fetching all announcements for a specific symbol
    """
    
    client = CodalClient()
    
    # Safely print symbol (handle Unicode issues)
    try:
        print(f"\nFetching all announcements for symbol: {symbol}")
    except UnicodeEncodeError:
        print(f"\nFetching all announcements for symbol: [Persian text]")
    
    # Search by symbol with date range
    announcements = client.search_by_symbol(
        symbol=symbol,
        from_date=YEAR_RANGES[1402][0],
        to_date=YEAR_RANGES[1402][1]
    )
    
    try:
        print(f"Found {len(announcements)} announcements for {symbol}")
    except UnicodeEncodeError:
        print(f"Found {len(announcements)} announcements for [Persian symbol]")
    
    # Process and analyze
    processor = DataProcessor(announcements)
    processor.add_letter_descriptions()
    
    # Group by letter code
    if not processor.df.empty:
        letter_distribution = processor.df['letter_code'].value_counts()
        print("\nAnnouncement types:")
        for code, count in letter_distribution.head(10).items():
            try:
                print(f"  {code}: {count}")
            except UnicodeEncodeError:
                # Skip problematic Unicode characters for display
                safe_code = str(code).encode('ascii', 'ignore').decode('ascii')
                print(f"  {safe_code}: {count}")
    
    return processor


def fetch_financial_statements_example():
    """
    Example of fetching financial statements
    """
    
    client = CodalClient()
    
    print("\nFetching annual financial statements...")
    
    # Search for annual audited financial statements
    statements = client.search_financial_statements(
        from_date=YEAR_RANGES[1401][0],
        to_date=YEAR_RANGES[1402][1],
        period_length=12,  # Annual reports
        audited_only=True  # Only audited statements
    )
    
    print(f"Found {len(statements)} financial statements")
    
    # Process the data
    processor = DataProcessor(statements)
    
    # Aggregate by symbol and year
    aggregated = processor.aggregate_by_symbol_year()
    
    print(f"\nAggregated to {len(aggregated)} symbol-year observations")
    
    # Export
    output_path = Path('output') / 'financial_statements.xlsx'
    processor.to_excel(output_path)
    print(f"Data exported to: {output_path}")
    
    return processor


def custom_query_example():
    """
    Example of building a custom query
    """
    
    client = CodalClient()
    
    print("\nBuilding custom query...")
    
    # Build a complex custom query
    try:
        results = (client
                   .set_letter_code('ن-45')  # Board changes
                   .set_company_type('1')     # Main exchange
                   .set_date_range(YEAR_RANGES[1400][0], CURRENT_DATE)
                   .set_entity_type(include_childs=False, include_mains=True)
                   .set_audit_status(audited=True, not_audited=True)
                   .fetch_all_pages(max_pages=5))  # Limit to 5 pages for example
    except Exception as e:
        print(f"Custom query failed: {e}")
        # Fallback to simpler query
        results = (client
                   .set_letter_code('ن-45')  # Board changes
                   .set_company_type('1')     # Main exchange
                   .fetch_all_pages(max_pages=2))
    
    print(f"Custom query returned {len(results)} results")
    
    # Get query statistics
    stats = client.get_summary_stats()
    print(f"Total available results: {stats['total_results']}")
    print(f"Total pages: {stats['total_pages']}")
    
    return results


if __name__ == "__main__":
    # Run examples
    print("=" * 60)
    print("CODAL SCRAPER EXAMPLES")
    print("=" * 60)
    
    # Example 1: Fetch board changes
    board_processor = fetch_board_changes_example()
    
    # Example 2: Fetch for specific symbol
    try:
        symbol_processor = fetch_specific_symbol_example('فولاد')
    except UnicodeEncodeError:
        # Fallback to ASCII-safe symbol if Unicode fails
        symbol_processor = fetch_specific_symbol_example('FOLD')
    
    # Example 3: Fetch financial statements
    # financial_processor = fetch_financial_statements_example()
    
    # Example 4: Custom query
    custom_results = custom_query_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)