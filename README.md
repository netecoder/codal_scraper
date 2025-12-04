## Codal Scraper

High-level Python toolkit for fetching and processing disclosure data from `Codal.ir`, the official information system for Iranian listed companies.

The library provides:

- **Synchronous client** (`CodalClient`) for Codal's search API
- **Asynchronous client** (`AsyncCodalClient`) for high‑throughput scraping
- **Data processing pipeline** (`DataProcessor`) for cleaning, filtering, aggregating and exporting results
- **Board member scraper** (`BoardMemberScraper`) for detailed هیئت‌مدیره information using Playwright/Crawlee
- **Utilities** for Persian dates, text normalization, caching, and rate limiting

The codebase is structured as a reusable package (`src/codal_scraper`) with example scripts in `examples/` and tests in `tests/`.

---

## Installation

### From source (this repo)

```bash
git clone https://github.com/<your-username>/codal_scraper.git
cd codal_scraper
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

If you later publish to PyPI, users will be able to install with:

```bash
pip install codal_scraper
```

### Optional extras

Some features are optional and require extra dependencies:

- **Async board scraping** (`BoardMemberScraper`): `crawlee[playwright]`, `playwright`
- **Network visualization**: `pyvis`

You can install everything (for development) with:

```bash
pip install -r requirements.txt
```

---

## Quick start

### Synchronous: search Codal and export to Excel

```python
from pathlib import Path
from codal_scraper import CodalClient, DataProcessor

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Search for هیئت‌مدیره changes (letter code ن-45) in a date range
with CodalClient() as client:
    letters = client.search_board_changes(
        from_date="1402/01/01",
        to_date="1402/12/29",
        max_pages=5,
    )

processor = DataProcessor(letters)
processor.add_letter_descriptions()
processor.to_excel(output_dir / "board_changes.xlsx")
```

The `examples/basic_usage.py` script contains more end‑to‑end demos:

- Search board changes
- Search by symbol
- Fetch financial statements
- Chain multiple processing steps and export to CSV/Excel/JSON/Parquet

### Asynchronous: high‑throughput fetching

```python
import asyncio
from pathlib import Path
from codal_scraper import AsyncCodalClient, DataProcessor

async def main():
    async with AsyncCodalClient(max_concurrent=5) as client:
        letters = await client.fetch_board_changes(
            from_date="1402/01/01",
            to_date="1402/12/29",
            max_pages=20,
        )

    processor = DataProcessor(letters)
    processor.add_letter_descriptions()
    processor.to_excel(Path("output") / "async_board_changes.xlsx")

asyncio.run(main())
```

See `examples/async_usage.py` for:

- Basic async fetching
- Fetching multiple symbols concurrently
- Progress reporting
- Comparing sync vs async performance

### Scraping detailed board member data (هیئت‌مدیره)

```python
import asyncio
from pathlib import Path
from codal_scraper import CodalClient, BoardMemberScraper

async def main():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 1) Get board‑change announcements and URLs
    with CodalClient() as client:
        letters = client.search_board_changes(
            from_date="1402/06/01",
            to_date="1402/12/29",
            max_pages=5,
        )
        urls = client.extract_letter_urls(letters)

    urls = urls[:10]  # small demo subset

    # 2) Scrape board member details from each Codal page
    async with BoardMemberScraper(max_concurrent=3, headless=True) as scraper:
        df = await scraper.scrape_urls(urls)
        scraper.export_to_excel(df, output_dir / "board_members.xlsx")

asyncio.run(main())
```

`examples/board_members_example.py` shows a full workflow:

- Fetch ن‑45 announcements
- Save Codal URLs to CSV
- Scrape detailed board‑member records
- Export data, errors, and network statistics
- Build an interactive board‑network visualization

---

## Main components

### `CodalClient` (sync)

Located in `src/codal_scraper/client.py`, exposed as `codal_scraper.CodalClient`.

- **Fluent, chainable API** based on Codal’s query parameters
- **Retry logic**, **rate limiting** and **file‑based caching**
- Convenience methods:
  - `search_board_changes(from_date, to_date, company_type=None, max_pages=None)`
  - `search_financial_statements(from_date, to_date, period_length=12, audited_only=True, max_pages=None)`
  - `search_monthly_reports(from_date, to_date, symbol=None, max_pages=None)`
  - `search_by_symbol(symbol, from_date=None, to_date=None, max_pages=None)`
- Helpers:
  - `extract_letter_urls(letters)`, `save_urls_to_csv(letters, "letter_urls.csv")`
  - `download_financial_excel_files(...)`
  - `get_summary_stats()`, `get_pagination_info()`

### `AsyncCodalClient`

Located in `src/codal_scraper/async_client.py`, exposed as `codal_scraper.AsyncCodalClient`.

- Uses `aiohttp` with an async rate limiter
- Fetches multiple pages concurrently via `fetch_all_pages`
- High‑level methods mirror the sync client:
  - `fetch_board_changes(...)`
  - `fetch_financial_statements(...)`
  - `fetch_monthly_reports(...)`
  - `fetch_by_symbol(...)`
  - `fetch_multiple_symbols(symbols, from_date, to_date, ...)`
- `get_stats()` returns a `QueryStats` dataclass with performance metrics

### `DataProcessor`

Located in `src/codal_scraper/processor.py`, exposed as `codal_scraper.DataProcessor`.

- Wraps a pandas `DataFrame` created from Codal API `Letters`
- Normalizes column names to `snake_case`
- Common operations (all chainable):
  - Filtering: `filter_by_letter_code`, `filter_by_date_range`, `filter_by_symbols`, `filter_by_condition`
  - Cleaning: `remove_duplicates`, `fill_na`, `drop_columns`, `reset_index`
  - Transformations: `add_letter_descriptions`, `add_column`, `apply_to_column`, `rename_columns`, `select_columns`, `sort_by`
  - Aggregation: `aggregate_by_symbol_year`, `group_by`
- Export:
  - `to_excel(path, include_summary=True)`
  - `to_csv(path)`
  - `to_json(path)`
  - `to_parquet(path)`
  - `to_dataframe()`, `to_dict()`
- Loading helpers:
  - `from_json_file`, `from_excel_file`, `from_csv_file`, `from_parquet_file`

### `BoardMemberScraper`

Located in `src/codal_scraper/board_scraper.py`, exposed as `codal_scraper.BoardMemberScraper`.

- Uses `crawlee` + Playwright to open each Codal ن‑45 page and parse the هیئت‌مدیره grid
- Extracts a rich `BoardMemberData` record per board row (member identity, position, independence, education, multiple‑role flags, CEO info, etc.)
- Maintains:
  - `members_data` list
  - `networkx` graph of company ↔ person relations
  - Structured error list with `ScrapingErrorInfo`
- High‑level methods:
  - `scrape_urls(urls_or_df, url_column="url")`
  - `scrape_from_csv(csv_path, url_column="url")`
  - `export_to_excel(...)`, `export_to_csv(...)`, `export_errors(...)`
  - `visualize_network("board_network.html")`
  - `get_stats()`, `get_error_summary()`, `reset()`

---

## Utilities and supporting modules

- `codal_scraper.constants` – API URLs, header templates, Codal letter‑code mappings, default rate‑limit/cache settings, Board‑member CSS selectors, Persian year utilities.
- `codal_scraper.types` – `TypedDict` and `dataclass` types such as `LetterData`, `BoardMemberData`, `DateRange`, `QueryStats`, `PaginationInfo`, `ScrapingErrorInfo`.
- `codal_scraper.utils` – helpers for:
  - Persian/Arabic digit conversion and text normalization
  - Date conversion between Gregorian and Jalali (Shamsi)
  - Safe dict operations, URL building, list/collection utilities
  - Converting API responses to pandas `DataFrame` (`parse_codal_response`)
- `codal_scraper.cache` – file‑based (`FileCache`) and in‑memory (`MemoryCache`) caching with TTL.
- `codal_scraper.rate_limiter` – sync (`RateLimiter`, `AdaptiveRateLimiter`) and async (`AsyncRateLimiter`) token‑bucket limiters.
- `codal_scraper.validators` – `InputValidator` and `validate_input` for strict checking of Persian dates, symbols, ISIC codes, URLs, etc.
- `codal_scraper.exceptions` – well‑structured exception hierarchy (`CodalScraperError`, `ValidationError`, `APIError`, `RateLimitError`, `NetworkError`, `ParseError`, `ScrapingError`, `CacheError`, `ConfigurationError`).

---

## Working with dates and the Persian calendar

Codal uses the **Jalali (Shamsi) calendar** with dates formatted as `YYYY/MM/DD` (e.g. `1402/01/01`).

This library:

- Validates Persian dates via `InputValidator.is_date`
- Offers helpers:
  - `get_current_persian_date()`, `get_current_persian_year()` (in `constants`)
  - `gregorian_to_shamsi`, `shamsi_to_gregorian`, `datetime_to_num`, `num_to_datetime` (in `utils`)
- Provides `DateRange` dataclass for strongly‑typed date ranges

Always pass Persian dates like `"1402/01/01"` to query methods unless you explicitly convert them yourself.

---

## Error handling

Most public methods raise custom exceptions instead of raw ones:

- **Input problems** → `ValidationError`
- **HTTP / API issues** → `APIError`, `NetworkError`, `RateLimitError`
- **Parsing issues** → `ParseError`
- **Scraping problems** → `ScrapingError` and detailed `ScrapingErrorInfo` records

Example pattern:

```python
from codal_scraper import CodalClient
from codal_scraper.exceptions import ValidationError, APIError, NetworkError

try:
    with CodalClient() as client:
        letters = client.search_board_changes("1402/01/01", "1402/12/29")
except ValidationError as e:
    print("Bad input:", e)
except APIError as e:
    print("Codal API error:", e)
except NetworkError as e:
    print("Network issue:", e)
```

---

## Project layout

- `src/codal_scraper/` – main package
- `examples/` – runnable scripts demonstrating sync, async and board‑scraper workflows
- `tests/` – pytest tests for client, cache, processor, validators, utils
- `output/` – example output files (Excel, CSV, JSON) created by sample scripts
- `storage/` – Crawlee key‑value stores and request queues created during board scraping (can be safely removed/ignored)

---

## How to run the examples

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python examples/basic_usage.py
python examples/async_usage.py
python examples/board_members_example.py
```

Most examples write results under the `output/` directory.

---

## Packaging and publishing

This repository already includes:

- `src/` layout with `codal_scraper` package
- `setup.py` and `pyproject.toml` (build configuration)
- `requirements.txt` for development

To build and publish to PyPI (once you have an account and have updated metadata such as author email and GitHub URL):

```bash
pip install build twine
python -m build
twine upload dist/*
```

Update `pyproject.toml` and `setup.py` with your **GitHub URL**, **author email**, and any classifiers you want before publishing.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.


