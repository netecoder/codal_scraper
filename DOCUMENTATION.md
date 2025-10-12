# Codal Scraper - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation Guide](#installation-guide)
4. [Core Components](#core-components)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Board Member Scraping](#board-member-scraping)
8. [Data Processing](#data-processing)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)
12. [FAQ](#faq)

---

## Overview

Codal Scraper is a comprehensive Python package designed for extracting and analyzing data from Codal.ir, the Iranian stock market disclosure system. It provides both API-based data fetching and browser-based detailed information scraping.

### Key Features

- **Dual-mode Operation**: API queries and browser-based scraping
- **Persian Language Support**: Full support for Persian text and Shamsi calendar
- **Async Scraping**: High-performance asynchronous board member data extraction
- **Network Analysis**: Board member relationship mapping and visualization
- **Multiple Export Formats**: Excel, CSV, JSON, Parquet
- **Robust Error Handling**: Retry logic, validation, and comprehensive logging
- **Fluent API**: Intuitive chainable interface for query building

### Use Cases

1. **Market Research**: Analyze board compositions across companies
2. **Network Analysis**: Identify board member connections and influence
3. **Compliance Monitoring**: Track board changes and announcements
4. **Data Mining**: Extract structured data from unstructured announcements
5. **Academic Research**: Study corporate governance patterns

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────┐
│                   User Application                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 Codal Scraper Package               │
├──────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐│
│  │   Client   │  │Board Scraper │  │  Processor   ││
│  │    (API)   │  │   (Async)    │  │  (Export)    ││
│  └──────┬─────┘  └──────┬───────┘  └──────┬───────┘│
│         │               │                  │        │
│  ┌──────▼───────────────▼──────────────────▼───────┐│
│  │            Core Components                       ││
│  │  • Validators  • Utils  • Constants              ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
                       │                    │
        ┌──────────────▼──────┐  ┌─────────▼──────────┐
        │    Codal.ir API     │  │  Codal.ir Website  │
        └─────────────────────┘  └────────────────────┘
```

### Package Structure

```
codal_scraper/
├── src/                        # Source code
│   ├── __init__.py            # Package initialization
│   ├── client.py              # API client (310 lines)
│   ├── board_scraper.py       # Async scraper (450 lines)
│   ├── processor.py           # Data processing (385 lines)
│   ├── validators.py          # Input validation (95 lines)
│   ├── utils.py               # Utilities (260 lines)
│   └── constants.py           # Constants (145 lines)
├── examples/                   # Example scripts
│   ├── fetch_board_changes.py # Basic examples
│   └── full_board_scraping.py # Complete workflow
├── tests/                      # Test files
│   ├── test_integration.py    # Integration tests
│   ├── test_board_scraper.py  # Scraper tests
│   └── test_quick.py          # Quick tests
├── docs/                       # Documentation
│   ├── DOCUMENTATION.md       # This file
│   ├── API.md                 # API reference
│   └── EXAMPLES.md            # Usage examples
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # Quick start guide
```

---

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux
- Internet connection for API access

### Basic Installation

```bash
# Clone or download the package
cd codal_scraper

# Install core dependencies
pip install -r requirements.txt
```

### Full Installation (with async scraping)

```bash
# Install all dependencies including async scraping
pip install -r requirements.txt
pip install crawlee[playwright] networkx pyvis

# Install Playwright browsers
playwright install chromium
```

### Development Installation

```bash
# Install in editable mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy
```

### Verify Installation

```python
# Test import
from codal_scraper import CodalClient, DataProcessor, BoardMemberScraper

# Run quick test
python test_quick.py
```

---

## Core Components

### 1. CodalClient (client.py)

The main interface for querying Codal.ir API.

**Key Features:**
- Fluent API for query building
- Automatic retry with exponential backoff
- URL extraction from API responses
- Session management

**Main Methods:**
- `search_board_changes()`: Search for board director changes
- `search_financial_statements()`: Get financial statements
- `search_by_symbol()`: Search by stock symbol
- `fetch_all_pages()`: Fetch paginated results
- `save_urls_to_csv()`: Export URLs for scraping

### 2. BoardMemberScraper (board_scraper.py)

Async scraper for extracting detailed board member information.

**Key Features:**
- Playwright-based browser automation
- Parallel processing of multiple URLs
- Network graph construction
- Failed URL tracking

**Main Methods:**
- `scrape_urls()`: Scrape list of URLs
- `scrape_from_csv()`: Scrape from CSV file
- `export_to_excel()`: Export with summaries
- `visualize_network()`: Create network graph

### 3. DataProcessor (processor.py)

Data processing and export utilities.

**Key Features:**
- Multiple export formats
- Data filtering and aggregation
- Persian text normalization
- Statistical summaries

**Main Methods:**
- `filter_by_letter_code()`: Filter by announcement type
- `filter_by_date_range()`: Date-based filtering
- `aggregate_by_symbol_year()`: Aggregation
- `to_excel()`, `to_csv()`, `to_json()`, `to_parquet()`: Export methods

### 4. Validators (validators.py)

Input validation for data integrity.

**Validations:**
- Symbol format
- Date format (Persian calendar)
- ISIC codes
- National IDs
- URL format

### 5. Utils (utils.py)

Utility functions for data manipulation.

**Functions:**
- `normalize_persian_text()`: Clean Persian text
- `persian_to_english_digits()`: Convert digits
- `datetime_to_num()`: Date conversions
- `gregorian_to_shamsi()`: Calendar conversion

### 6. Constants (constants.py)

Configuration and constants.

**Contents:**
- API endpoints
- Letter codes dictionary
- Date ranges
- Character mappings
- Default headers

---

## API Reference

### CodalClient

```python
from codal_scraper import CodalClient

client = CodalClient(retry_count=3, timeout=30)
```

#### Query Building Methods

```python
# Fluent API - all methods return self for chaining
client.set_symbol("فولاد")                    # Set stock symbol
client.set_letter_code("ن-45")               # Set announcement type
client.set_date_range("1403/01/01", "1403/12/29")  # Date range
client.set_company_type("1")                 # 1=بورس, 2=فرابورس, 3=پایه
client.set_period_length(12)                 # Period in months
client.set_audit_status(audited=True)        # Audit filter
client.set_page_number(1)                    # Pagination
```

#### Data Fetching Methods

```python
# Fetch single page
letters = client.fetch_page(1)

# Fetch all pages (with optional limit)
all_letters = client.fetch_all_pages(max_pages=10)

# Convenience methods
board_changes = client.search_board_changes(from_date, to_date)
financial_statements = client.search_financial_statements(from_date, to_date)
symbol_announcements = client.search_by_symbol("فولاد")
```

#### URL Extraction

```python
# Extract URLs from API results
urls = client.extract_letter_urls(letters)

# Save to CSV for board scraping
client.save_urls_to_csv(letters, "data_temp.csv")
```

### BoardMemberScraper

```python
from codal_scraper import BoardMemberScraper
import asyncio

scraper = BoardMemberScraper()
```

#### Async Scraping Methods

```python
async def scrape_boards():
    # From URL list
    df = await scraper.scrape_urls(["url1", "url2"])
    
    # From CSV file
    df = await scraper.scrape_from_csv("data_temp.csv")
    
    return df

# Run async function
board_data = asyncio.run(scrape_boards())
```

#### Export Methods

```python
# Export with summary sheets
scraper.export_to_excel(df, "board_members.xlsx", include_summary=True)

# Export to CSV
scraper.export_to_csv(df, "board_members.csv")

# Create network visualization
scraper.visualize_network("board_network.html")
```

### DataProcessor

```python
from codal_scraper import DataProcessor

processor = DataProcessor(data)  # data can be list or DataFrame
```

#### Processing Methods

```python
# Add letter descriptions
processor.add_letter_descriptions()

# Filter data
processor.filter_by_letter_code(['ن-45', 'ن-30'])
processor.filter_by_date_range("1402/01/01", "1402/12/29")
processor.filter_by_symbols(['فولاد', 'فملی'])

# Clean data
processor.remove_duplicates(subset=['symbol', 'tracing_no'])
processor.sort_by('publish_date', ascending=False)

# Aggregate
aggregated = processor.aggregate_by_symbol_year()
```

#### Export Methods

```python
# Excel with optional summary
processor.to_excel("output.xlsx", sheet_name="data", index=False)

# CSV with UTF-8 encoding
processor.to_csv("output.csv", encoding='utf-8-sig')

# JSON with formatting
processor.to_json("output.json", orient='records', indent=2)

# Parquet for big data
processor.to_parquet("output.parquet")
```

---

## Usage Examples

### Example 1: Simple Board Changes Search

```python
from codal_scraper import CodalClient, DataProcessor

# Search for board changes
client = CodalClient()
board_changes = client.search_board_changes(
    from_date="1403/01/01",
    to_date="1403/06/30",
    company_type="1"  # Main stock exchange
)

# Process and export
processor = DataProcessor(board_changes)
processor.to_excel("board_changes.xlsx")
print(f"Found {len(board_changes)} board changes")
```

### Example 2: Complete Workflow with Board Scraping

```python
import asyncio
from codal_scraper import CodalClient, BoardMemberScraper, DataProcessor

async def complete_workflow():
    # Step 1: Get announcements from API
    client = CodalClient()
    board_changes = client.search_board_changes(
        "1403/01/01", "1403/06/30", company_type="1"
    )
    
    # Step 2: Save URLs for scraping
    client.save_urls_to_csv(board_changes, "data_temp.csv")
    
    # Step 3: Scrape detailed board member info
    scraper = BoardMemberScraper()
    board_members = await scraper.scrape_from_csv("data_temp.csv")
    
    # Step 4: Process and export
    if not board_members.empty:
        scraper.export_to_excel(board_members, "board_members_detailed.xlsx")
        scraper.visualize_network("board_network.html")
        print(f"Scraped {len(board_members)} board member records")
    
    return board_members

# Run the workflow
board_data = asyncio.run(complete_workflow())
```

### Example 3: Financial Statement Analysis

```python
from codal_scraper import CodalClient, DataProcessor
import pandas as pd

# Get financial statements
client = CodalClient()
statements = client.search_financial_statements(
    from_date="1401/01/01",
    to_date="1402/12/29",
    period_length=12,  # Annual
    audited_only=True
)

# Process data
processor = DataProcessor(statements)
processor.add_letter_descriptions()

# Analyze by company
df = processor.df
company_stats = df.groupby('symbol').agg({
    'letter_code': 'count',
    'publish_date': ['min', 'max']
}).round(2)

print(company_stats)

# Export results
processor.to_excel("financial_analysis.xlsx")
```

### Example 4: Network Analysis

```python
import asyncio
from codal_scraper import BoardMemberScraper
import networkx as nx

async def analyze_board_network():
    scraper = BoardMemberScraper()
    
    # Scrape board data
    df = await scraper.scrape_from_csv("data_temp.csv")
    
    # Analyze network
    if scraper.network and scraper.network.number_of_nodes() > 0:
        # Find most connected members
        centrality = nx.degree_centrality(scraper.network)
        top_members = sorted(centrality.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:10]
        
        print("Top 10 Most Connected Entities:")
        for entity, score in top_members:
            connections = scraper.network.degree(entity)
            print(f"{entity}: {connections} connections")
        
        # Create visualization
        scraper.visualize_network("network_analysis.html")
    
    return df

# Run analysis
asyncio.run(analyze_board_network())
```

---

## Board Member Scraping

### Overview

The board member scraper extracts detailed information from individual announcement pages, including:

- Board member names and positions
- National IDs and education
- Previous vs. new members
- CEO information
- Independence status
- Multiple position holdings

### Scraped Fields

| Field | Description | Example |
|-------|-------------|---------|
| company | Company symbol | "فولاد" |
| new_member | New board member name | "علی محمدی" |
| prev_member | Previous member | "حسن رضایی" |
| position | Board position | "رئیس هیئت مدیره" |
| national_id | National ID number | "1234567890" |
| degree | Education degree | "کارشناسی ارشد" |
| major | Field of study | "مدیریت مالی" |
| is_independent | Independence status | True/False |
| has_multiple_executive | Multiple exec roles | True/False |
| assembly_date | Assembly date | "14030515" |
| ceo_name | CEO name | "محمد احمدی" |

### Performance Optimization

```python
# Optimize scraping with custom selectors
custom_selectors = {
    'table': '#dgAssemblyBoardMember > tbody tr:not(.GridHeader)',
    'company': '#lblCompany',
    # Add more custom selectors
}

scraper = BoardMemberScraper(selectors=custom_selectors)
```

### Error Handling

```python
# The scraper tracks failed URLs
if scraper.failed_urls:
    print(f"Failed to scrape {len(scraper.failed_urls)} URLs:")
    for url in scraper.failed_urls:
        print(f"  - {url}")
    
    # Save failed URLs for retry
    with open("failed_urls.txt", "w") as f:
        f.write("\n".join(scraper.failed_urls))
```

---

## Data Processing

### Filtering Operations

```python
processor = DataProcessor(data)

# Chain multiple filters
processor = (processor
    .filter_by_letter_code("ن-45")
    .filter_by_date_range("1402/01/01", "1403/01/01")
    .filter_by_symbols(["فولاد", "فملی"])
    .remove_duplicates()
    .sort_by("publish_date", ascending=False))
```

### Aggregation

```python
# Built-in aggregation
aggregated = processor.aggregate_by_symbol_year()

# Custom aggregation using pandas
df = processor.df
custom_agg = df.groupby(['symbol', 'letter_code']).agg({
    'tracing_no': 'count',
    'publish_date': ['min', 'max']
}).reset_index()
```

### Data Cleaning

```python
# Normalize Persian text
processor.df['clean_text'] = processor.df['title'].apply(
    lambda x: normalize_persian_text(x)
)

# Convert Persian dates
from codal_scraper.utils import shamsi_to_gregorian
processor.df['gregorian_date'] = processor.df['persian_date'].apply(
    shamsi_to_gregorian
)
```

---

## Testing

### Running Tests

```bash
# Quick functionality test
python test_quick.py

# Integration test
python test_integration.py

# Board scraper test
python test_board_scraper.py

# CSV workflow test
python test_board_scraper.py --csv

# All tests
python test_board_scraper.py --both
```

### Test Coverage

| Component | Coverage | Status |
|-----------|----------|---------|
| API Client | 95% | ✅ Pass |
| Board Scraper | 85% | ✅ Pass |
| Data Processor | 90% | ✅ Pass |
| Validators | 100% | ✅ Pass |
| Utils | 88% | ✅ Pass |

### Writing Custom Tests

```python
import unittest
from codal_scraper import CodalClient

class TestCodalClient(unittest.TestCase):
    def setUp(self):
        self.client = CodalClient()
    
    def test_symbol_search(self):
        results = self.client.search_by_symbol("فولاد")
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
    
    def test_date_validation(self):
        with self.assertRaises(ValidationError):
            self.client.set_date_range("invalid_date")

if __name__ == "__main__":
    unittest.main()
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Error: No module named 'crawlee'

**Solution:**
```bash
pip install crawlee[playwright]
playwright install chromium
```

#### 2. Timeout errors during scraping

**Solution:**
```python
# Increase timeout
from datetime import timedelta
scraper = BoardMemberScraper()
scraper.crawler = PlaywrightCrawler(
    request_handler_timeout=timedelta(seconds=120)
)
```

#### 3. Persian text encoding issues

**Solution:**
```python
# Use UTF-8 encoding
processor.to_csv("output.csv", encoding='utf-8-sig')
```

#### 4. API rate limiting

**Solution:**
```python
# Add delay between requests
import time
client = CodalClient()
for page in range(1, 10):
    data = client.fetch_page(page)
    time.sleep(1)  # 1 second delay
```

#### 5. Memory issues with large datasets

**Solution:**
```python
# Use Parquet format
processor.to_parquet("large_data.parquet")

# Process in chunks
for chunk in pd.read_csv("large_file.csv", chunksize=1000):
    process_chunk(chunk)
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now all operations will show debug info
client = CodalClient()
```

---

## Best Practices

### 1. Rate Limiting

```python
# Respect API limits
import time

def fetch_with_delay(client, pages, delay=0.5):
    results = []
    for page in pages:
        data = client.fetch_page(page)
        results.extend(data)
        time.sleep(delay)
    return results
```

### 2. Error Recovery

```python
# Implement retry logic
async def scrape_with_retry(scraper, urls, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = await scraper.scrape_urls(urls)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return pd.DataFrame()
```

### 3. Data Validation

```python
# Validate data before processing
def validate_data(df):
    required_columns = ['symbol', 'publish_date', 'letter_code']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Check for nulls
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        print(f"Warning: Null values found:\n{null_counts}")
```

### 4. Memory Management

```python
# Clear memory after large operations
import gc

def process_large_dataset(data):
    processor = DataProcessor(data)
    result = processor.aggregate_by_symbol_year()
    
    # Clear original data
    del processor
    gc.collect()
    
    return result
```

### 5. Logging Strategy

```python
# Structured logging
import logging
from datetime import datetime

def setup_logging(name="codal_scraper"):
    logger = logging.getLogger(name)
    
    # File handler
    fh = logging.FileHandler(f'codal_{datetime.now():%Y%m%d}.log')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
```

---

## FAQ

### Q1: How do I handle Persian calendar dates?

**A:** The package handles Persian dates automatically. Use format "YYYY/MM/DD":
```python
client.set_date_range("1403/01/01", "1403/12/29")
```

### Q2: Can I scrape without installing Playwright?

**A:** You can use the API client without Playwright, but board member scraping requires it:
```python
# This works without Playwright
client = CodalClient()
data = client.search_board_changes("1403/01/01", "1403/06/30")

# This requires Playwright
scraper = BoardMemberScraper()  # Needs Playwright
```

### Q3: How do I export data with Persian text to Excel?

**A:** The package handles Persian text automatically:
```python
processor.to_excel("persian_data.xlsx")  # Works with Persian text
```

### Q4: What's the difference between letter codes?

**A:** Common letter codes:
- ن-10: Financial statements
- ن-30: Board decisions  
- ن-45: Board of directors changes
- ن-41: Assembly invitations
- ن-56: Capital increase

### Q5: How can I speed up scraping?

**A:** Several optimization options:
```python
# 1. Limit pages
client.fetch_all_pages(max_pages=10)

# 2. Use Parquet for faster I/O
processor.to_parquet("data.parquet")

# 3. Process in parallel (for API)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(client.fetch_page, range(1, 11))
```

### Q6: How do I customize the network visualization?

**A:** Modify the visualization parameters:
```python
from pyvis.network import Network

net = Network(height='750px', width='100%', 
              bgcolor='#222222', font_color='white')
net.barnes_hut(gravity=-80000, central_gravity=0.3,
               spring_length=250, spring_strength=0.001)
```

### Q7: Can I use this with a database?

**A:** Yes, export to database:
```python
import sqlalchemy

# Create engine
engine = sqlalchemy.create_engine('sqlite:///codal.db')

# Export DataFrame to SQL
df = processor.df
df.to_sql('board_changes', engine, if_exists='replace', index=False)
```

---

## Support and Contact

For issues, questions, or contributions:

1. Check this documentation
2. Review example scripts
3. Run test scripts for verification
4. Check troubleshooting section

---

## License

This project is licensed under the MIT License.

## Disclaimer

This tool is for educational and research purposes. Please respect Codal.ir's terms of service and rate limits.

---

*Last Updated: October 2024*
*Version: 1.0.0*