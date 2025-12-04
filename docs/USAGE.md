## Codal Scraper – Detailed Usage & Architecture

This document provides a deeper look at how the library is organized and how to use its main components effectively.

---

## High‑level architecture

- **`CodalClient`** (`src/codal_scraper/client.py`)
  - Sync HTTP client built on `requests`
  - Encapsulates Codal search API parameters and pagination
  - Integrates **file cache** (`FileCache`) and **rate limiting** (`RateLimiter`)
- **`AsyncCodalClient`** (`src/codal_scraper/async_client.py`)
  - Async HTTP client built on `aiohttp`
  - Concurrent page fetching with `asyncio.Semaphore` and `AsyncRateLimiter`
- **`DataProcessor`** (`src/codal_scraper/processor.py`)
  - DataFrame‑based processing layer for Codal `Letters`
  - Chainable transforms + multi‑format export
- **`BoardMemberScraper`** (`src/codal_scraper/board_scraper.py`)
  - Playwright/Crawlee‑based scraper for detailed board‑member (ن‑45) pages
  - Builds both a tabular dataset and a `networkx` graph
- **Infrastructure modules**
  - `constants.py` – URLs, headers, letter codes, default configs, selectors
  - `types.py` – enums, `TypedDict`s, dataclasses (e.g. `LetterData`, `BoardMemberData`, `DateRange`)
  - `utils.py` – text/date/URL helpers, response → DataFrame conversion, collection utilities
  - `cache.py` – `FileCache`, `MemoryCache` with TTL and stats
  - `rate_limiter.py` – `RateLimiter`, `AsyncRateLimiter`, `AdaptiveRateLimiter`
  - `validators.py` – `InputValidator` and helpers for Persian dates, symbols, etc.
  - `exceptions.py` – common error hierarchy

Everything is re‑exported from `codal_scraper.__init__` so typical users only import from `codal_scraper` itself.

---

## Using `CodalClient` (sync)

### Basic pattern

```python
from codal_scraper import CodalClient

with CodalClient() as client:
    letters = client.search_board_changes(
        from_date="1402/01/01",
        to_date="1402/12/29",
        max_pages=10,
    )
```

The client:

- Initializes lazily (`requests.Session` is created only when needed)
- Tracks statistics such as requests made, cache hits, and total items
- Supports a rich set of **parameter setters**, all chainable:

```python
letters = (
    CodalClient()
    .set_symbol("فولاد")
    .set_letter_code("ن-10")              # financial statements
    .set_date_range("1402/01/01", "1402/12/29")
    .set_period_length(12)
    .set_audit_status(audited=True, not_audited=False)
    .fetch_all_pages(max_pages=20)
)
```

### Parameters and validation

Most setters validate input and raise `ValidationError` on bad values:

- `set_symbol(symbol)` → validates Persian/ASCII ticker symbols
- `set_date_range(from_date, to_date)` → requires valid Jalali dates in `YYYY/MM/DD`
- `set_isic(isic)` → validates 4–6 digit ISIC codes
- `set_letter_code(code)` → expects patterns like `ن-45`

If you pass invalid dates such as a Gregorian year (`2024/01/01`), validation will fail early.

### Pagination helpers

- `fetch_page(page=1)` – fetch exactly one page and update pagination state
- `fetch_all_pages(max_pages=None)` – fetch the first page, infer total pages, then iterate over all (or up to `max_pages`)
- `get_pagination_info()` – returns a `PaginationInfo` dataclass with `current_page`, `total_pages`, `total_results`

### Caching and rate limiting

By default:

- Responses are cached on disk under `.codal_cache` (see `CacheConfig`)
- A token‑bucket rate limiter is used to avoid hammering Codal

You can disable these features or customize them:

```python
from codal_scraper import CodalClient, CacheConfig, RateLimitConfig

cache_cfg = CacheConfig(cache_dir=".codal_cache", default_ttl=3600)
rate_cfg = RateLimitConfig(requests_per_second=1.0, burst_limit=3)

client = CodalClient(
    cache_config=cache_cfg,
    rate_limit_config=rate_cfg,
    enable_cache=True,
    enable_rate_limit=True,
)
```

Use `client.clear_cache()` to invalidate all cached entries.

---

## Using `AsyncCodalClient`

### When to use

Use `AsyncCodalClient` when:

- You need to fetch many pages across long date ranges
- You are querying multiple symbols or multiple letter codes concurrently

It is `async` and should be used inside an event loop:

```python
import asyncio
from codal_scraper import AsyncCodalClient

async def main():
    async with AsyncCodalClient(max_concurrent=10) as client:
        letters = await client.fetch_financial_statements(
            from_date="1402/01/01",
            to_date="1402/12/29",
            period_length=12,
            audited_only=True,
            max_pages=50,
        )
    print(len(letters))

asyncio.run(main())
```

### Parallel symbol fetching

`fetch_multiple_symbols` wraps `fetch_by_symbol` in concurrent tasks:

```python
symbols = ["فولاد", "خودرو", "شپنا"]

async with AsyncCodalClient(max_concurrent=10) as client:
    results = await client.fetch_multiple_symbols(
        symbols=symbols,
        from_date="1402/01/01",
        to_date="1402/12/29",
        max_pages_per_symbol=5,
    )

for symbol, letters in results.items():
    print(symbol, len(letters))
```

### Statistics

After a run you can access `QueryStats`:

```python
stats = client.get_stats()
print(stats.total_results, stats.pages_fetched, stats.duration_seconds)
```

---

## Data processing with `DataProcessor`

`DataProcessor` is designed for interactive analysis and for pipelines.

### Initialization

```python
from codal_scraper import DataProcessor

processor = DataProcessor(letters)  # letters is a list of dicts from CodalClient
print(processor)                    # human‑readable summary
```

By default it:

- Converts column names (e.g. `PublishDateTime`) → `publish_date_time`
- Attempts to parse obvious date/time columns
- Normalizes common Persian text fields (`symbol`, `company_name`, etc.)

You can disable pieces of this behavior:

```python
processor = DataProcessor(
    letters,
    normalize_columns=True,
    process_dates=False,
    normalize_text=False,
)
```

### Common transformations

Filter by letter codes:

```python
processor.filter_by_letter_code(["ن-45", "ن-10"])
```

Filter by date range (using Persian date strings, not Gregorian):

```python
processor.filter_by_date_range(
    start_date="1402/01/01",
    end_date="1402/06/31",
    date_column="publish_date_time",  # or leave default
)
```

Filter by symbols:

```python
processor.filter_by_symbols(["فولاد", "فملی"])
```

Apply custom conditions:

```python
processor.filter_by_condition(
    lambda df: (df.get("has_excel", False) == True),
    description="has_excel=True",
)
```

Sort and remove duplicates:

```python
(processor
    .remove_duplicates(subset=["symbol", "tracing_no"])
    .sort_by("publish_date_time", ascending=False)
    .reset_index()
)
```

Select / drop / rename columns:

```python
processor.select_columns([
    "symbol", "company_name", "letter_code", "publish_date_time",
])

processor.rename_columns({
    "publish_date_time": "publish_dt",
})
```

### Aggregation and statistics

Quick symbol‑by‑year overview:

```python
summary_by_symbol_year = processor.aggregate_by_symbol_year()
print(summary_by_symbol_year.head())
```

Generic groupby:

```python
grouped = processor.group_by(
    columns="symbol",
    aggregations={"tracing_no": "count"},
)
```

General stats:

```python
stats = processor.get_summary_stats()
print(stats["total_records"], stats["unique_symbols"])
print(stats["letter_code_distribution"])
```

### Export

```python
from pathlib import Path

processor.to_excel(Path("output") / "letters.xlsx")
processor.to_csv("output/letters.csv")
processor.to_json("output/letters.json")
processor.to_parquet("output/letters.parquet")
```

The Excel export optionally includes:

- A **Summary** sheet with high‑level stats
- A **Column Info** sheet with per‑column metadata

---

## Board member scraping workflow

The board scraper is more advanced and relies on Playwright/Crawlee.

### 1. Get URLs (sync client)

```python
from codal_scraper import CodalClient

with CodalClient() as client:
    letters = client.search_board_changes(
        from_date="1402/06/01",
        to_date="1402/12/29",
        max_pages=5,
    )
    urls = client.extract_letter_urls(letters)
```

### 2. Scrape pages (Playwright)

```python
import asyncio
from codal_scraper import BoardMemberScraper

async def scrape_board_members(urls):
    async with BoardMemberScraper(
        max_retries=2,
        timeout=60,
        headless=True,
        max_concurrent=3,
    ) as scraper:
        df = await scraper.scrape_urls(urls)
        stats = scraper.get_stats()
        print("Members extracted:", stats["members_extracted"])
        return df, scraper

df, scraper = asyncio.run(scrape_board_members(urls[:10]))
```

### 3. Export and analyze

```python
from pathlib import Path

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

scraper.export_to_excel(df, output_dir / "board_members.xlsx")
scraper.export_to_csv(df, output_dir / "board_members.csv")

# Optional: network visualization
scraper.visualize_network(output_dir / "board_network.html")

# Inspect errors if any
if scraper.errors:
    scraper.export_errors(output_dir / "scraping_errors.json")
```

The resulting DataFrame contains all fields defined in `BoardMemberData` (see `types.py`), including:

- Company info, dates, and Codal URL
- Member identity and national ID (normalized digits)
- Position, independence flag, education
- Multiple‑role flags, corporate‑governance declarations
- CEO metadata and scrape timestamp

---

## Handling dates and localization

Codal’s API uses the Jalali calendar; this library assumes:

- All query dates are in `YYYY/MM/DD` **Persian** format (e.g., `1402/01/01`)
- `validators.InputValidator.is_date` enforces:
  - Year between 1300 and 1500
  - Correct month/day ranges, including leap‑year handling

Helpers in `utils.py`:

- `gregorian_to_shamsi("20230815")  -> "1402/05/24"`
- `shamsi_to_gregorian("1402/05/24") -> "20230815"`
- `datetime_to_num("1402/05/15 10:30:00") -> 14020515103000`
- `num_to_datetime(14020515103000) -> "1402/05/15 10:30:00"`

---

## Testing

Tests are written with `pytest` and live under `tests/`:

- `test_client.py` – client initialization, parameter setting, URL generation, basic fetching behavior
- `test_processor.py` – processor column normalization, filters, grouping, exports
- `test_cache.py`, `test_utils.py`, `test_validators.py` – focused unit tests for helpers

To run tests:

```bash
pip install -r requirements.txt
pytest
```

---

## Extending the library

Some ideas for extension:

- **New convenience queries**:
  - Wrap common Codal letter codes with high‑level methods (e.g., specific مجمع types)
- **Custom processors**:
  - Build thin wrappers around `DataProcessor` for domain‑specific analyses
- **Additional scrapers**:
  - Reuse the Playwright/Crawlee setup to scrape other Codal report types

When contributing:

- Add tests under `tests/`
- Keep public API imports in `__init__.py` in sync
- Update the `README.md` and this document with new features


