# Codal Scraper - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Advanced Usage](#advanced-usage)
6. [Data Processing](#data-processing)
7. [Async Web Scraping](#async-web-scraping)
8. [Configuration](#configuration)
9. [Error Handling](#error-handling)
10. [Performance Optimization](#performance-optimization)
11. [Examples](#examples)
12. [Troubleshooting](#troubleshooting)

## Overview

The Codal Scraper is a comprehensive Python package designed to interact with Codal.ir, the Iranian stock market disclosure system. It provides both synchronous API data fetching and asynchronous web scraping capabilities for extracting detailed financial and corporate information.

### Key Features

- **Dual-Mode Operation**: Synchronous API client and asynchronous web scraper
- **Persian Calendar Support**: Native support for Shamsi dates
- **Comprehensive Data Processing**: Built-in filtering, sorting, and normalization
- **Multiple Export Formats**: Excel, CSV, JSON, and Parquet
- **Network Analysis**: Board member relationship visualization
- **Robust Error Handling**: Retry logic and validation

## Installation

### Basic Installation

```bash
pip install codal-scraper
```

### With Optional Features

```bash
# Async web scraping capabilities
pip install codal-scraper[async]

# Network visualization features
pip install codal-scraper[network]

# Parquet export support
pip install codal-scraper[parquet]

# All optional features
pip install codal-scraper[all]

# Development dependencies
pip install codal-scraper[dev]
```

### Browser Installation (for async scraping)

```bash
playwright install chromium
```

## Quick Start

### Basic API Usage

```python
from codal_scraper import CodalClient, DataProcessor

# Initialize client
client = CodalClient()

# Fetch board changes
data = client.search_board_changes(
    from_date="1403/01/01",
    to_date="1403/12/29",
    company_type='1'
)

# Process data
processor = DataProcessor(data)
processor.to_excel("output.xlsx")
```

### Async Board Member Scraping

```python
import asyncio
from codal_scraper import CodalClient, BoardMemberScraper

async def main():
    # Get URLs from API
    client = CodalClient()
    data = client.search_board_changes("1403/01/01", "1403/06/30")
    client.save_urls_to_csv(data, 'urls.csv')
    
    # Scrape detailed info
    scraper = BoardMemberScraper()
    result = await scraper.scrape_from_csv('urls.csv')
    scraper.export_to_excel(result, 'board_members.xlsx')

asyncio.run(main())
```

## API Reference

### CodalClient

The main client for interacting with the Codal API.

#### Constructor

```python
CodalClient(retry_count: int = 3, timeout: int = 30)
```

**Parameters:**
- `retry_count`: Number of retry attempts for failed requests (default: 3)
- `timeout`: Request timeout in seconds (default: 30)

#### Core Methods

##### Parameter Setting Methods

All parameter setting methods return `self` for method chaining.

```python
# Symbol filtering
client.set_symbol(symbol: str) -> CodalClient

# Company type filtering
client.set_company_type(company_type: str) -> CodalClient
# Valid values: 'بورس', 'فرابورس', 'پایه', '1', '2', '3'

# Date range filtering
client.set_date_range(from_date: str = None, to_date: str = None) -> CodalClient
# Date format: "YYYY/MM/DD" (Persian calendar)

# Letter code filtering
client.set_letter_code(code: str) -> CodalClient
# Example: 'ن-45' for board changes

# Period length filtering
client.set_period_length(period: Union[int, str]) -> CodalClient
# Valid values: 1-12 (months)

# Audit status filtering
client.set_audit_status(audited: bool = True, not_audited: bool = True) -> CodalClient

# Consolidation status filtering
client.set_consolidation_status(consolidated: bool = True, not_consolidated: bool = True) -> CodalClient

# Entity type filtering
client.set_entity_type(include_childs: bool = True, include_mains: bool = True) -> CodalClient

# Year end date filtering
client.set_year_end(year_end_date: str) -> CodalClient

# Publisher filtering
client.set_publisher_only(publisher_only: bool = False) -> CodalClient

# Page number for pagination
client.set_page_number(page: int) -> CodalClient
```

##### Data Fetching Methods

```python
# Fetch single page
client.fetch_page(page: int = None) -> Optional[List[Dict]]

# Fetch multiple pages
client.fetch_all_pages(max_pages: int = None) -> List[Dict]

# Get query URL
client.get_query_url(use_api: bool = True) -> str

# Get summary statistics
client.get_summary_stats() -> Dict
```

##### Convenience Methods

```python
# Search for board changes
client.search_board_changes(from_date: str, to_date: str, company_type: str = None) -> List[Dict]

# Search for financial statements
client.search_financial_statements(from_date: str, to_date: str, period_length: int = 12, audited_only: bool = True) -> List[Dict]

# Search by symbol
client.search_by_symbol(symbol: str, from_date: str = None, to_date: str = None) -> List[Dict]

# Extract URLs from API response
client.extract_letter_urls(letters: List[Dict]) -> List[str]

# Save URLs to CSV
client.save_urls_to_csv(letters: List[Dict], file_path: str = 'data_temp.csv') -> None

# Download Excel files from financial statements
client.download_financial_excel_files(
    from_date: str,
    to_date: str,
    period_length: int = 12,
    audited_only: bool = True,
    output_dir: Union[str, Path] = "financial_excel",
    max_files: int = None
) -> List[str]

# Reset all parameters
client.reset_params() -> None
```

### DataProcessor

Process and export Codal data to various formats.

#### Constructor

```python
DataProcessor(data: Union[List[Dict], pd.DataFrame] = None)
```

#### Core Methods

##### Data Processing

```python
# Add letter code descriptions
processor.add_letter_descriptions() -> DataProcessor

# Filter by letter codes
processor.filter_by_letter_code(codes: Union[str, List[str]]) -> DataProcessor

# Filter by date range
processor.filter_by_date_range(start_date: str = None, end_date: str = None, date_column: str = 'publish_date') -> DataProcessor

# Filter by symbols
processor.filter_by_symbols(symbols: Union[str, List[str]]) -> DataProcessor

# Remove duplicates
processor.remove_duplicates(subset: List[str] = None, keep: str = 'last') -> DataProcessor

# Sort data
processor.sort_by(columns: Union[str, List[str]], ascending: Union[bool, List[bool]] = True) -> DataProcessor

# Aggregate by symbol and year
processor.aggregate_by_symbol_year() -> pd.DataFrame

# Get summary statistics
processor.get_summary_stats() -> Dict
```

##### Export Methods

```python
# Export to Excel
processor.to_excel(filepath: Union[str, Path], sheet_name: str = 'data', index: bool = False) -> None

# Export to CSV
processor.to_csv(filepath: Union[str, Path], index: bool = False, encoding: str = 'utf-8') -> None

# Export to JSON
processor.to_json(filepath: Union[str, Path], orient: str = 'records', indent: int = 2) -> None

# Export to Parquet (requires parquet extra)
processor.to_parquet(filepath: Union[str, Path]) -> None
```

##### Class Methods

```python
# Load from JSON file
DataProcessor.from_json_file(filepath: Union[str, Path]) -> DataProcessor

# Load from Excel file
DataProcessor.from_excel_file(filepath: Union[str, Path], sheet_name: Union[str, int] = 0) -> DataProcessor
```

### BoardMemberScraper

Async scraper for extracting detailed board member information.

#### Constructor

```python
BoardMemberScraper(selectors: Dict[str, str] = None)
```

**Parameters:**
- `selectors`: Custom CSS selectors for extracting information

#### Core Methods

```python
# Scrape from URL list
await scraper.scrape_urls(urls: Union[List[str], pd.DataFrame]) -> pd.DataFrame

# Scrape from CSV file
await scraper.scrape_from_csv(csv_path: str) -> pd.DataFrame

# Export to Excel
scraper.export_to_excel(df: pd.DataFrame = None, file_path: str = 'board_members.xlsx', include_summary: bool = True) -> None

# Export to CSV
scraper.export_to_csv(df: pd.DataFrame = None, file_path: str = 'board_members.csv') -> None

# Create network visualization
scraper.visualize_network(output_file: str = 'board_network.html') -> None
```

## Advanced Usage

### Complex Query Building

```python
# Build sophisticated queries using method chaining
results = (client
    .set_letter_code('ن-45')                    # Board changes
    .set_company_type('1')                      # Main exchange
    .set_date_range("1400/01/01", "1403/12/29")
    .set_entity_type(include_childs=False, include_mains=True)
    .set_audit_status(audited=True, not_audited=False)
    .set_consolidation_status(consolidated=True, not_consolidated=False)
    .fetch_all_pages(max_pages=50))
```

### Batch Processing

```python
# Process multiple symbols
symbols = ['فولاد', 'فملی', 'شبندر']
all_data = []

for symbol in symbols:
    data = client.search_by_symbol(symbol, "1402/01/01", "1402/12/29")
    all_data.extend(data)

# Combine and process
processor = DataProcessor(all_data)
processor.remove_duplicates()
processor.to_excel("combined_data.xlsx")
```

### Custom Data Processing

```python
# Load existing data and apply custom filters
processor = DataProcessor.from_excel_file("existing_data.xlsx")

# Apply complex filtering
df = processor.df
filtered = df[
    (df['letter_code'] == 'ن-45') &
    (df['symbol'].str.contains('فولاد')) &
    (df['publish_date'] >= '1403-01-01')
]

# Create new processor with filtered data
new_processor = DataProcessor(filtered)
new_processor.to_excel("filtered_data.xlsx")
```

## Data Processing

### Column Normalization

The DataProcessor automatically:
- Converts column names to snake_case
- Normalizes Persian text fields
- Handles Persian date formats
- Processes datetime columns

### Text Processing

```python
from codal_scraper.utils import normalize_persian_text, persian_to_english_digits

# Normalize Persian text
text = normalize_persian_text("متن فارسی")
# Converts Arabic letters to Persian, removes zero-width characters

# Convert Persian digits to English
digits = persian_to_english_digits("۱۲۳۴۵")
# Returns: "12345"
```

### Date Handling

```python
from codal_scraper.utils import gregorian_to_shamsi, shamsi_to_gregorian

# Convert Gregorian to Persian
persian_date = gregorian_to_shamsi("20240101")  # Returns: "1402/10/11"

# Convert Persian to Gregorian
gregorian_date = shamsi_to_gregorian("1402/10/11")  # Returns: "20240101"
```

## Async Web Scraping

### Basic Async Scraping

```python
import asyncio
from codal_scraper import BoardMemberScraper

async def scrape_board_data():
    scraper = BoardMemberScraper()
    
    # URLs to scrape
    urls = [
        "https://codal.ir/Reports/Decision.aspx?LetterSerial=...",
        "https://codal.ir/Reports/Decision.aspx?LetterSerial=..."
    ]
    
    # Scrape data
    df = await scraper.scrape_urls(urls)
    
    # Export results
    scraper.export_to_excel(df, 'board_members.xlsx')
    
    return df

# Run the async function
result = asyncio.run(scrape_board_data())
```

### CSV-Based Workflow

```python
async def csv_workflow():
    # Step 1: Get URLs from API
    client = CodalClient()
    board_changes = client.search_board_changes(
        from_date="1403/01/01",
        to_date="1403/06/30",
        company_type='1'
    )
    
    # Step 2: Save URLs to CSV
    client.save_urls_to_csv(board_changes, 'board_urls.csv')
    
    # Step 3: Scrape detailed information
    scraper = BoardMemberScraper()
    detailed_data = await scraper.scrape_from_csv('board_urls.csv')
    
    # Step 4: Export and analyze
    scraper.export_to_excel(detailed_data, 'detailed_board_data.xlsx')
    
    # Create network visualization
    if not detailed_data.empty:
        scraper.visualize_network('board_network.html')
    
    return detailed_data
```

### Custom Selectors

```python
# Custom CSS selectors for different page layouts
custom_selectors = {
    'table': '.custom-board-table tr',
    'company': '.company-name',
    'ceo_name': '.ceo-info',
    # ... other selectors
}

scraper = BoardMemberScraper(selectors=custom_selectors)
```

## Downloading Excel Files from Financial Statements

### Basic Excel Download

```python
from codal_scraper import CodalClient

client = CodalClient()

# Download Excel files from financial statements
downloaded_files = client.download_financial_excel_files(
    from_date="1401/01/01",
    to_date="1402/12/29",
    period_length=12,      # Annual reports
    audited_only=True,     # Only audited statements
    output_dir="financial_excel",
    max_files=50           # Limit to 50 files
)

print(f"Downloaded {len(downloaded_files)} Excel files")
```

### Advanced Excel Download Workflow

```python
def download_and_analyze_financial_excel():
    """Download Excel files and analyze them"""
    
    client = CodalClient()
    
    # Step 1: Download Excel files
    downloaded_files = client.download_financial_excel_files(
        from_date="1401/01/01",
        to_date="1402/12/29",
        period_length=12,
        audited_only=True,
        output_dir="financial_excel",
        max_files=10  # Limit for testing
    )
    
    # Step 2: Load and analyze the Excel files
    import pandas as pd
    
    all_data = []
    for file_path in downloaded_files:
        try:
            df = pd.read_excel(file_path)
            all_data.append(df)
            print(f"Loaded: {file_path} ({len(df)} rows)")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
    
    # Step 3: Combine and analyze
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total rows: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")
    
    return downloaded_files

# Run the workflow
files = download_and_analyze_financial_excel()
```

## Configuration

### Logging Configuration

```python
import logging

# Configure logging level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or configure specific logger
logger = logging.getLogger('codal_scraper')
logger.setLevel(logging.DEBUG)
```

### Custom Headers

```python
from codal_scraper.constants import DEFAULT_HEADERS

# Modify default headers
client.session.headers.update({
    'User-Agent': 'Custom User Agent',
    'Accept-Language': 'fa-IR,fa;q=0.9,en-US;q=0.8'
})
```

### Timeout and Retry Configuration

```python
# Custom timeout and retry settings
client = CodalClient(retry_count=5, timeout=60)

# Or modify after initialization
client.retry_count = 5
client.timeout = 60
```

## Error Handling

### Validation Errors

```python
from codal_scraper.validators import ValidationError

try:
    client.set_symbol("invalid@symbol")
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Request Errors

```python
try:
    data = client.fetch_page(1)
    if data is None:
        print("Failed to fetch data")
except Exception as e:
    print(f"Request failed: {e}")
```

### Async Scraping Errors

```python
async def safe_scraping():
    scraper = BoardMemberScraper()
    
    try:
        df = await scraper.scrape_urls(urls)
        if df.empty:
            print("No data scraped")
        else:
            print(f"Successfully scraped {len(df)} records")
    except Exception as e:
        print(f"Scraping failed: {e}")
        print(f"Failed URLs: {scraper.failed_urls}")
```

## Performance Optimization

### Caching Strategies

```python
import json
from pathlib import Path

def cache_api_data(data, filename):
    """Cache API data to avoid repeated requests"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_cached_data(filename):
    """Load cached API data"""
    if Path(filename).exists():
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Usage
cache_file = 'board_changes_cache.json'
cached_data = load_cached_data(cache_file)

if cached_data is None:
    data = client.search_board_changes("1403/01/01", "1403/12/29")
    cache_api_data(data, cache_file)
else:
    data = cached_data
```

### Batch Processing

```python
def process_symbols_in_batches(symbols, batch_size=10):
    """Process symbols in batches to avoid overwhelming the API"""
    all_data = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {batch}")
        
        for symbol in batch:
            try:
                data = client.search_by_symbol(symbol, "1403/01/01", "1403/06/30")
                all_data.extend(data)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Failed to process {symbol}: {e}")
    
    return all_data
```

### Memory Optimization

```python
# For large datasets, use Parquet format
processor.to_parquet("large_dataset.parquet")

# Load only when needed
def process_large_dataset():
    df = pd.read_parquet("large_dataset.parquet")
    # Process in chunks
    for chunk in pd.read_parquet("large_dataset.parquet", chunksize=1000):
        # Process chunk
        pass
```

## Examples

### Complete Board Analysis Workflow

```python
import asyncio
from pathlib import Path
from codal_scraper import CodalClient, DataProcessor, BoardMemberScraper

async def complete_board_analysis():
    """Complete workflow for board member analysis"""
    
    # Step 1: Fetch board changes from API
    print("📊 Fetching board changes from API...")
    client = CodalClient()
    board_changes = client.search_board_changes(
        from_date="1403/01/01",
        to_date="1403/06/30",
        company_type='1'
    )
    
    # Step 2: Process API data
    print("🔄 Processing API data...")
    processor = DataProcessor(board_changes)
    processor.add_letter_descriptions()
    
    # Get statistics
    stats = processor.get_summary_stats()
    print(f"Found {stats['total_records']} board changes")
    print(f"Unique companies: {stats['unique_symbols']}")
    
    # Step 3: Export API data
    print("💾 Exporting API data...")
    processor.to_excel("api_board_changes.xlsx")
    
    # Step 4: Extract URLs for detailed scraping
    print("🔗 Extracting URLs...")
    urls = client.extract_letter_urls(board_changes[:10])  # Limit for demo
    print(f"Extracted {len(urls)} URLs")
    
    # Step 5: Scrape detailed board information
    print("🕷️ Scraping detailed board information...")
    scraper = BoardMemberScraper()
    detailed_data = await scraper.scrape_urls(urls)
    
    if not detailed_data.empty:
        print(f"Scraped {len(detailed_data)} detailed records")
        
        # Step 6: Export detailed data
        print("💾 Exporting detailed data...")
        scraper.export_to_excel(detailed_data, "detailed_board_data.xlsx")
        
        # Step 7: Create network visualization
        print("🕸️ Creating network visualization...")
        scraper.visualize_network("board_network.html")
        
        # Step 8: Analysis
        print("📈 Analysis:")
        print(f"  - Unique companies: {detailed_data['company'].nunique()}")
        print(f"  - Unique board members: {detailed_data['new_member'].nunique()}")
        
        if 'is_independent' in detailed_data.columns:
            independent_pct = (detailed_data['is_independent'].sum() / len(detailed_data)) * 100
            print(f"  - Independent members: {independent_pct:.1f}%")
    
    print("✅ Analysis complete!")
    return detailed_data

# Run the analysis
result = asyncio.run(complete_board_analysis())
```

### Financial Statement Analysis

```python
def analyze_financial_statements():
    """Analyze financial statements over multiple years"""
    
    client = CodalClient()
    all_statements = []
    
    # Fetch statements for multiple years
    years = [1401, 1402, 1403]
    
    for year in years:
        print(f"Fetching statements for year {year}...")
        statements = client.search_financial_statements(
            from_date=f"{year}/01/01",
            to_date=f"{year}/12/29",
            period_length=12,  # Annual
            audited_only=True
        )
        all_statements.extend(statements)
    
    # Process all statements
    processor = DataProcessor(all_statements)
    processor.add_letter_descriptions()
    
    # Aggregate by symbol and year
    aggregated = processor.aggregate_by_symbol_year()
    
    # Export results
    processor.to_excel("financial_statements.xlsx")
    aggregated.to_excel("financial_aggregated.xlsx", index=False)
    
    print(f"Processed {len(all_statements)} financial statements")
    print(f"Aggregated to {len(aggregated)} symbol-year observations")
    
    return processor, aggregated

# Run the analysis
processor, aggregated = analyze_financial_statements()
```

### Custom Query Builder

```python
def build_custom_queries():
    """Demonstrate custom query building"""
    
    client = CodalClient()
    
    # Query 1: Board changes for specific companies
    query1 = (client
        .set_letter_code('ن-45')
        .set_company_type('1')
        .set_date_range("1403/01/01", "1403/06/30")
        .fetch_all_pages(max_pages=5))
    
    # Query 2: Financial statements for specific period
    query2 = (client
        .set_letter_code('ن-10')
        .set_period_length(12)
        .set_audit_status(audited=True, not_audited=False)
        .set_date_range("1402/01/01", "1402/12/29")
        .fetch_all_pages(max_pages=10))
    
    # Query 3: All announcements for specific symbol
    query3 = (client
        .set_symbol('فولاد')
        .set_date_range("1403/01/01", "1403/06/30")
        .fetch_all_pages())
    
    # Process and export each query
    results = [
        ("Board Changes", query1),
        ("Financial Statements", query2),
        ("Foulad Announcements", query3)
    ]
    
    for name, data in results:
        if data:
            processor = DataProcessor(data)
            processor.add_letter_descriptions()
            filename = f"{name.lower().replace(' ', '_')}.xlsx"
            processor.to_excel(filename)
            print(f"Exported {len(data)} records to {filename}")
    
    return results

# Run the custom queries
results = build_custom_queries()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'codal_scraper'`

**Solutions**:
```bash
# For development installation
pip install -e .

# For regular installation
pip install codal-scraper

# Check if package is installed
pip list | grep codal-scraper
```

#### 2. Async Dependencies Missing

**Problem**: `ImportError: No module named 'crawlee'`

**Solution**:
```bash
pip install codal-scraper[async]
```

#### 3. Playwright Browser Issues

**Problem**: `Error: Browser not found` or `playwright._impl._api_types.Error: Browser not found`

**Solutions**:
```bash
# Install Playwright browsers
playwright install chromium

# Install all browsers
playwright install

# For Windows, you might need to run as administrator
```

#### 4. Persian Text Display Issues

**Problem**: Persian text appears as question marks or boxes

**Solutions**:
- Ensure your terminal supports UTF-8
- Use a Unicode-compatible font
- Set environment variable: `PYTHONIOENCODING=utf-8`

#### 5. Date Format Issues

**Problem**: Date validation errors

**Solutions**:
```python
# Use Persian calendar dates
client.set_date_range("1403/01/01", "1403/12/29")  # ✅ Correct

# Not Gregorian dates
client.set_date_range("2024/01/01", "2024/12/29")  # ❌ Incorrect
```

#### 6. Rate Limiting Issues

**Problem**: Too many requests or timeouts

**Solutions**:
```python
# Increase timeout and retry count
client = CodalClient(retry_count=5, timeout=60)

# Add delays between requests
import time
time.sleep(2)  # 2 second delay

# Use max_pages to limit requests
data = client.fetch_all_pages(max_pages=5)
```

#### 7. Memory Issues with Large Datasets

**Problem**: Out of memory when processing large datasets

**Solutions**:
```python
# Use Parquet format instead of Excel
processor.to_parquet("large_data.parquet")

# Process in chunks
for chunk in pd.read_parquet("large_data.parquet", chunksize=1000):
    # Process each chunk
    pass

# Use data types to reduce memory usage
df = df.astype({'column': 'category'})
```

#### 8. Network Issues

**Problem**: Connection timeouts or network errors

**Solutions**:
```python
# Increase timeout
client = CodalClient(timeout=120)

# Use proxy if needed
client.session.proxies = {
    'http': 'http://proxy:port',
    'https': 'https://proxy:port'
}

# Check network connectivity
import requests
try:
    response = requests.get("https://codal.ir", timeout=10)
    print("Network connectivity OK")
except:
    print("Network connectivity issues")
```

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or enable for specific components
logging.getLogger('codal_scraper').setLevel(logging.DEBUG)
logging.getLogger('codal_scraper.client').setLevel(logging.DEBUG)
```

### Performance Monitoring

Monitor performance and identify bottlenecks:

```python
import time
import psutil

def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        print(f"Function: {func.__name__}")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        print(f"Memory usage: {end_memory - start_memory:.2f} MB")
        
        return result
    return wrapper

# Usage
@monitor_performance
def fetch_large_dataset():
    client = CodalClient()
    return client.fetch_all_pages(max_pages=20)

data = fetch_large_dataset()
```

---

This documentation provides comprehensive guidance for using the Codal Scraper package. For additional support, please refer to the GitHub repository or create an issue.