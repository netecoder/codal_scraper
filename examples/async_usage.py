#!/usr/bin/env python
"""
Async usage example for Codal Scraper

This example demonstrates how to use the AsyncCodalClient for
high-performance concurrent data fetching.
"""

import asyncio
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_async():
    """Basic async fetching example"""
    from codal_scraper import AsyncCodalClient, DataProcessor
    
    logger.info("Example 1: Basic async fetching")
    
    start_time = time.time()
    
    async with AsyncCodalClient(max_concurrent=5) as client:
        # Fetch board changes
        letters = await client.fetch_board_changes(
            from_date="1402/01/01",
            to_date="1402/12/29",
            max_pages=20
        )
        
        # Get stats
        stats = client.get_stats()
        logger.info(f"Stats: {stats}")
    
    duration = time.time() - start_time
    logger.info(f"Fetched {len(letters)} letters in {duration:.2f}s")
    
    # Process the data
    if letters:
        processor = DataProcessor(letters)
        processor.add_letter_descriptions()
        processor.to_excel(Path("output") / "async_board_changes.xlsx")
    
    return letters


async def example_multiple_symbols():
    """Fetch data for multiple symbols concurrently"""
    from codal_scraper import AsyncCodalClient, DataProcessor
    
    logger.info("\nExample 2: Fetching multiple symbols concurrently")
    
    symbols = ["فولاد", "خودرو", "شپنا", "فملی", "کگل"]
    
    start_time = time.time()
    
    async with AsyncCodalClient(max_concurrent=10) as client:
        results = await client.fetch_multiple_symbols(
            symbols=symbols,
            from_date="1402/01/01",
            to_date="1402/12/29",
            max_pages_per_symbol=5
        )
    
    duration = time.time() - start_time
    
    # Print results
    for symbol, letters in results.items():
        logger.info(f"  {symbol}: {len(letters)} announcements")
    
    logger.info(f"Total time: {duration:.2f}s")
    
    # Combine all results
    all_letters = []
    for letters in results.values():
        all_letters.extend(letters)
    
    if all_letters:
        processor = DataProcessor(all_letters)
        processor.add_letter_descriptions()
        processor.to_excel(Path("output") / "multiple_symbols.xlsx")
    
    return results


async def example_with_progress():
    """Fetch with progress callback"""
    from codal_scraper import AsyncCodalClient
    
    logger.info("\nExample 3: Fetching with progress tracking")
    
    def progress_callback(current: int, total: int):
        percent = current / total * 100
        bar_length = 30
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\rProgress: [{bar}] {percent:.1f}% ({current}/{total})", end='', flush=True)
    
    async with AsyncCodalClient() as client:
        params = {
            "LetterCode": "ن-30",  # Monthly reports
            "FromDate": "1402/01/01",
            "ToDate": "1402/12/29",
            "Audited": "true",
            "NotAudited": "true",
            "Consolidatable": "true",
            "NotConsolidatable": "true",
            "Childs": "true",
            "Mains": "true",
        }
        
        letters = await client.fetch_all_pages(
            params,
            max_pages=30,
            progress_callback=progress_callback
        )
    
    print()  # New line after progress bar
    logger.info(f"Fetched {len(letters)} monthly reports")
    
    return letters


async def example_compare_sync_async():
    """Compare performance of sync vs async"""
    from codal_scraper import CodalClient, AsyncCodalClient
    
    logger.info("\nExample 4: Comparing sync vs async performance")
    
    # Parameters
    from_date = "1402/01/01"
    to_date = "1402/06/31"
    max_pages = 10
    
    # Sync timing
    logger.info("Running sync version...")
    sync_start = time.time()
    
    with CodalClient() as client:
        sync_letters = client.search_board_changes(
            from_date=from_date,
            to_date=to_date,
            max_pages=max_pages
        )
    
    sync_duration = time.time() - sync_start
    
    # Async timing
    logger.info("Running async version...")
    async_start = time.time()
    
    async with AsyncCodalClient(max_concurrent=5) as client:
        async_letters = await client.fetch_board_changes(
            from_date=from_date,
            to_date=to_date,
            max_pages=max_pages
        )
    
    async_duration = time.time() - async_start
    
    # Results
    logger.info(f"\nResults:")
    logger.info(f"  Sync:  {len(sync_letters)} letters in {sync_duration:.2f}s")
    logger.info(f"  Async: {len(async_letters)} letters in {async_duration:.2f}s")
    logger.info(f"  Speedup: {sync_duration/async_duration:.2f}x")


async def example_error_handling():
    """Example with error handling"""
    from codal_scraper import AsyncCodalClient
    from codal_scraper.exceptions import APIError, RateLimitError
    
    logger.info("\nExample 5: Error handling in async")
    
    try:
        async with AsyncCodalClient(
            max_concurrent=10,
            retry_count=3,
            timeout=30
        ) as client:
            letters = await client.fetch_board_changes(
                from_date="1402/01/01",
                to_date="1402/12/29",
                max_pages=5
            )
            
            logger.info(f"Successfully fetched {len(letters)} letters")
            return letters
            
    except RateLimitError as e:
        logger.error(f"Rate limited: {e}")
        logger.info(f"Retry after: {e.retry_after}s")
        
    except APIError as e:
        logger.error(f"API error: {e}")
        logger.info(f"Status code: {e.status_code}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    return []


async def main():
    """Run all async examples"""
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Run examples
    await example_basic_async()
    await example_multiple_symbols()
    await example_with_progress()
    await example_compare_sync_async()
    await example_error_handling()
    
    logger.info("\nAll async examples completed!")


if __name__ == "__main__":
    asyncio.run(main())