# Codal Scraper

A comprehensive Python package for scraping and analyzing data from Codal.ir, the Iranian stock market disclosure system. This package provides both synchronous API data fetching and asynchronous web scraping capabilities for detailed board member information.

## 🚀 Features

- **📊 API Data Fetching**: Robust synchronous client for Codal API data retrieval
- **🕷️ Async Web Scraping**: Advanced board member detail extraction using Playwright
- **🔄 Fluent API Interface**: Chainable methods for building complex queries
- **📈 Data Processing**: Built-in processing and normalization for Persian text
- **💾 Multiple Export Formats**: Excel, CSV, JSON, and Parquet support
- **🕸️ Network Visualization**: Interactive board member network graphs
- **✅ Input Validation**: Comprehensive validation for all parameters
- **🗓️ Persian Calendar Support**: Native support for Shamsi (Persian) dates
- **⚡ Retry Logic**: Automatic retry with exponential backoff
- **📦 Modular Design**: Install only what you need

## 📋 Requirements

- Python 3.8+
- Internet connection for API access

## 🔧 Installation

### Basic Installation (Core Features)

```bash
pip install codal-scraper
```

### With Optional Features

```bash
# For async web scraping (board member details)
pip install codal-scraper[async]

# For network visualization
pip install codal-scraper[network]

# For Parquet export support
pip install codal-scraper[parquet]

# For all optional features
pip install codal-scraper[all]
```

### Development Installation

```bash
git clone https://github.com/yourusername/codal-scraper.git
cd codal-scraper
pip install -e .
```

### Install Playwright Browsers (for async scraping)

If you're using the async scraping features, you need to install Playwright browsers:

```bash
playwright install chromium
```

## 🚀 Quick Start

### Basic Usage - API Data Fetching

```python
from codal_scraper import CodalClient, DataProcessor

# Initialize client
client = CodalClient()

# Search for board of directors changes
board_changes = client.search_board_changes(
    from_date="1403/01/01",
    to_date="1403/12/29",
    company_type='1'  # Main stock exchange
)

# Process and export data
processor = DataProcessor(board_changes)
processor.to_excel("board_changes.xlsx")
print(f"Found {len(board_changes)} board change announcements")
```

### Advanced Usage - Board Member Scraping

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
    
    # Step 2: Save URLs to CSV for processing
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

## 📚 Usage Examples

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

### 4. Data Processing and Analysis

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
processor.to_parquet("output.parquet")  # Requires parquet extra
```

## 🔧 Configuration

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

## 📊 Letter Codes Reference

Common letter codes and their meanings:

| Code | Description |
|------|-------------|
| **ن-10** | Financial statements |
| **ن-30** | Board decisions |
| **ن-45** | Board of directors changes |
| **ن-41** | Assembly invitations |
| **ن-56** | Capital increase |
| **ن-58** | Important information |

For a complete list, see the [Letter Codes Documentation](DOCUMENTATION.md#letter-codes).

## 🏗️ Architecture

### Core Components

- **CodalClient**: Synchronous API client for data fetching
- **BoardMemberScraper**: Asynchronous web scraper for detailed information
- **DataProcessor**: Data processing, filtering, and export utilities
- **InputValidator**: Input validation and sanitization
- **Utils**: Helper functions for text processing and date conversion

### Data Flow

```
API Data → CodalClient → DataProcessor → Export Formats
    ↓
URLs → BoardMemberScraper → Detailed Data → Network Analysis
```

## 🧪 Testing

Run the included tests to verify your installation:

```bash
# Basic functionality test
python tests/test_quick.py

# Integration test
python tests/test_integration.py

# Board scraper test (requires async dependencies)
python tests/test_board_scraper.py
```

## ⚠️ Error Handling

The package includes comprehensive error handling:

```python
from codal_scraper.validators import ValidationError

try:
    client.set_symbol("invalid@symbol")
except ValidationError as e:
    print(f"Validation error: {e}")
```

## 🚨 Rate Limiting and Best Practices

- **Respect Rate Limits**: The package includes automatic delays between requests
- **Use Appropriate Timeouts**: Default timeout is 30 seconds
- **Limit Page Fetching**: Use `max_pages` parameter during testing
- **Cache Results**: Save fetched data locally to avoid repeated API calls

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
ModuleNotFoundError: No module named 'codal_scraper'
```
**Solution**: Install the package properly:
```bash
pip install -e .  # For development
# or
pip install codal-scraper
```

#### 2. Async Dependencies Missing
```bash
ImportError: No module named 'crawlee'
```
**Solution**: Install async dependencies:
```bash
pip install codal-scraper[async]
```

#### 3. Playwright Browser Not Found
```bash
Error: Browser not found
```
**Solution**: Install Playwright browsers:
```bash
playwright install chromium
```

#### 4. Persian Text Display Issues
**Solution**: Ensure your terminal/IDE supports UTF-8 encoding.

#### 5. Date Format Issues
**Solution**: Use Persian calendar dates in YYYY/MM/DD format:
```python
# Correct
client.set_date_range("1403/01/01", "1403/12/29")

# Incorrect
client.set_date_range("2024/01/01", "2024/12/29")
```

### Performance Tips

1. **Use Parquet for Large Datasets**: Much faster than Excel for large files
2. **Limit Page Fetching**: Use `max_pages` parameter during development
3. **Cache Results**: Save API data locally to avoid repeated requests
4. **Batch Processing**: Process multiple symbols in batches

## 📈 Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Fetch 1 page (20 items) | ~1-2 seconds | API dependent |
| Process 1000 records | ~0.1 seconds | Local processing |
| Export to Excel | ~2-5 seconds | File size dependent |
| Export to Parquet | ~0.5 seconds | Much faster than Excel |
| Scrape 10 board pages | ~30-60 seconds | Network dependent |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/codal-scraper.git`
3. Install in development mode: `pip install -e .[dev]`
4. Install pre-commit hooks: `pre-commit install`
5. Make your changes and add tests
6. Run tests: `pytest`
7. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all public methods
- Include tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Disclaimer

This tool is for educational and research purposes. Please respect Codal.ir's terms of service and rate limits when using this scraper. The authors are not responsible for any misuse of this tool.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/codal-scraper/issues)
- **Documentation**: [Full Documentation](DOCUMENTATION.md)
- **Examples**: See the `examples/` directory

## 🔄 Changelog

### Version 1.0.0
- ✅ Fixed import path issues
- ✅ Fixed datetime_to_num function bugs
- ✅ Improved date filtering for Persian dates
- ✅ Fixed URL generation issues
- ✅ Added comprehensive optional dependencies
- ✅ Improved error handling and validation
- ✅ Added extensive documentation and examples

## 🙏 Acknowledgments

- Codal.ir for providing the data API
- The Playwright team for the excellent browser automation framework
- The pandas and requests library maintainers

---

**Made with ❤️ for the Iranian financial data community**