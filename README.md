# Codal Scraper

A comprehensive Python package for scraping and analyzing data from Codal.ir, the Iranian stock market disclosure system. Now with integrated board member detailed information scraping!

## Features

- **Fluent API Interface**: Chainable methods for building complex queries
- **Board Member Scraping**: Async scraping of detailed board member information from Codal pages
- **Robust Data Fetching**: Automatic retry logic and error handling
- **Data Processing**: Built-in processing and normalization for Persian text
- **Multiple Export Formats**: Excel, CSV, JSON, and Parquet
- **Network Visualization**: Interactive board member network graphs
- **Validation**: Comprehensive input validation for all parameters
- **Caching Support**: Efficient data caching to reduce API calls
- **Persian Calendar Support**: Native support for Shamsi (Persian) dates

## Installation

### Requirements

```bash
pip install requests pandas openpyxl jdatetime
```

### For Board Member Scraping (Async)

```bash
pip install crawlee[playwright] networkx pyvis
playwright install chromium
```

### Optional (for Parquet support)

```bash
pip install pyarrow
```

## Quick Start

```python
from codal_scraper import CodalClient, DataProcessor

# Initialize client
client = CodalClient()

# Search for board of directors changes
board_changes = client.search_board_changes(
    from_date="1402/01/01",
    to_date="1403/12/29",
    company_type='1'  # Main stock exchange
)

# Process and export data
processor = DataProcessor(board_changes)
processor.to_excel("board_changes.xlsx")
```

## Usage Examples

### 1. Search by Symbol

```python
# Get all announcements for a specific symbol
announcements = client.search_by_symbol(
    symbol="فولاد",
    from_date="1402/01/01",
    to_date="1402/12/29"
)
```

### 2. Get Financial Statements

```python
# Fetch annual audited financial statements
statements = client.search_financial_statements(
    from_date="1401/01/01",
    to_date="1402/12/29",
    period_length=12,      # Annual reports
    audited_only=True      # Only audited statements
)
```

### 3. Custom Query Builder

```python
# Build complex queries using fluent interface
results = (client
    .set_letter_code('ن-45')              # Board changes
    .set_company_type('1')                 # Main exchange
    .set_date_range("1400/01/01", "1403/12/29")
    .set_entity_type(include_childs=False, include_mains=True)
    .set_audit_status(audited=True, not_audited=False)
    .fetch_all_pages(max_pages=10))
```

### 4. Board Member Scraping (Async)

```python
import asyncio
from codal_scraper import CodalClient, BoardMemberScraper

async def scrape_board_members():
    # Step 1: Get board change announcements from API
    client = CodalClient()
    board_changes = client.search_board_changes(
        from_date="1403/01/01",
        to_date="1403/06/30",
        company_type='1'
    )
    
    # Step 2: Save URLs to CSV
    client.save_urls_to_csv(board_changes, 'data_temp.csv')
    
    # Step 3: Scrape detailed board member info
    scraper = BoardMemberScraper()
    board_members_df = await scraper.scrape_from_csv('data_temp.csv')
    
    # Step 4: Export and visualize
    scraper.export_to_excel(board_members_df, 'board_members.xlsx')
    scraper.visualize_network('board_network.html')
    
    return board_members_df

# Run the async function
board_data = asyncio.run(scrape_board_members())
```

### 5. Data Processing

```python
processor = DataProcessor(results)

# Add descriptions for letter codes
processor.add_letter_descriptions()

# Filter by specific criteria
processor.filter_by_letter_code(['ن-45', 'ن-30'])
processor.filter_by_date_range(start_date="1402/01/01")

# Remove duplicates and sort
processor.remove_duplicates(subset=['symbol', 'tracing_no'])
processor.sort_by('publish_date', ascending=False)

# Export to different formats
processor.to_excel("output.xlsx")
processor.to_csv("output.csv")
processor.to_json("output.json")
processor.to_parquet("output.parquet")
```

## API Reference

### CodalClient

Main client for interacting with Codal API.

#### Methods

- `set_symbol(symbol)`: Set stock symbol for query
- `set_company_type(type)`: Set company type ('بورس', 'فرابورس', 'پایه' or '1', '2', '3')
- `set_isic(code)`: Set ISIC industry code
- `set_subject(subject)`: Set announcement subject
- `set_tracing_number(number)`: Set tracing number
- `set_letter_code(code)`: Set letter code (e.g., 'ن-45')
- `set_period_length(months)`: Set reporting period (1-12 months)
- `set_date_range(from_date, to_date)`: Set date range (Persian calendar)
- `set_audit_status(audited, not_audited)`: Set audit status filters
- `set_consolidation_status(consolidated, not_consolidated)`: Set consolidation filters
- `set_entity_type(include_childs, include_mains)`: Set entity type filters
- `set_year_end(date)`: Set fiscal year end date
- `set_publisher_only(bool)`: Filter organization-published announcements
- `fetch_page(page)`: Fetch a single page of results
- `fetch_all_pages(max_pages)`: Fetch multiple pages
- `get_query_url(use_api)`: Generate query URL
- `get_summary_stats()`: Get query statistics

### DataProcessor

Process and export Codal data.

#### Methods

- `add_letter_descriptions()`: Add human-readable descriptions for letter codes
- `filter_by_letter_code(codes)`: Filter by letter code(s)
- `filter_by_date_range(start, end, column)`: Filter by date range
- `filter_by_symbols(symbols)`: Filter by stock symbol(s)
- `remove_duplicates(subset, keep)`: Remove duplicate rows
- `sort_by(columns, ascending)`: Sort data by column(s)
- `aggregate_by_symbol_year()`: Aggregate data by symbol and year
- `get_summary_stats()`: Get summary statistics
- `to_excel(filepath, sheet_name, index)`: Export to Excel
- `to_csv(filepath, index, encoding)`: Export to CSV
- `to_json(filepath, orient, indent)`: Export to JSON
- `to_parquet(filepath)`: Export to Parquet
- `from_json_file(filepath)`: Load from JSON file
- `from_excel_file(filepath, sheet)`: Load from Excel file

## Letter Codes

Common letter codes and their meanings:

- **ن-10**: Financial statements
- **ن-30**: Board decisions
- **ن-45**: Board of directors changes
- **ن-41**: Assembly invitations
- **ن-56**: Capital increase
- **ن-58**: Important information

## Error Handling

The package includes comprehensive error handling:

```python
from codal_scraper.validators import ValidationError

try:
    client.set_symbol("invalid@symbol")
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Configuration

### Logging

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.DEBUG)
```

### Custom Headers

```python
from codal_scraper.constants import DEFAULT_HEADERS

# Modify default headers if needed
client.session.headers.update({
    'User-Agent': 'Custom User Agent'
})
```

## Advanced Usage

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

### Custom Filtering

```python
# Load existing data and apply custom filters
processor = DataProcessor.from_excel_file("existing_data.xlsx")

# Apply complex filtering
df = processor.df
filtered = df[
    (df['letter_code'] == 'ن-45') &
    (df['symbol'].str.contains('فولاد')) &
    (df['publish_date'] >= '2023-01-01')
]

# Create new processor with filtered data
new_processor = DataProcessor(filtered)
new_processor.to_excel("filtered_data.xlsx")
```

## Performance Tips

1. **Use Parquet for large datasets**: Parquet files are much faster to read/write than Excel
2. **Limit page fetching**: Use `max_pages` parameter to limit API calls during testing
3. **Cache results**: Save fetched data locally to avoid repeated API calls
4. **Batch exports**: Process all data before exporting rather than exporting incrementally

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License.

## Disclaimer

This tool is for educational and research purposes. Please respect Codal.ir's terms of service and rate limits when using this scraper.

## Support

For issues, questions, or contributions, please open an issue on the project repository.