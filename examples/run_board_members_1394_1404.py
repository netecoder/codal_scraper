#!/usr/bin/env python
import asyncio
import logging
from pathlib import Path

import pandas as pd  # for df initialization & type

from codal_scraper import CodalClient, BoardMemberScraper, DataProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    # ====== CONFIG ======
    from_date = "1394/01/01"
    to_date = "1404/09/10"
    max_pages = 2000  # increase if needed

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # ====== Step 1: Search Codal for ن-45 (board changes) ======
    logger.info(f"Searching board changes from {from_date} to {to_date} ...")

    with CodalClient() as client:
        letters = client.search_board_changes(
            from_date=from_date,
            to_date=to_date,
            max_pages=max_pages,
        )
        logger.info(f"Found {len(letters)} announcements")

        urls = client.extract_letter_urls(letters)
        logger.info(f"Extracted {len(urls)} URLs")

        # Save URL list (not the full letters) for reference / re-runs
        client.save_urls_to_csv(
            letters,
            output_dir / "letters_board_changes_1394_1404.csv",
        )

    if not urls:
        logger.warning("No URLs found – nothing to scrape.")
        return

    # ====== Step 2: Scrape board member tables ======
    logger.info("Scraping board member details from URLs...")

    # Ensure df always exists even if scraping fails early
    df: pd.DataFrame = pd.DataFrame()

    async with BoardMemberScraper(
        max_retries=3,
        timeout=60,
        headless=True,      # explicit, even though it's the default
        max_concurrent=5,
    ) as scraper:
        # IMPORTANT: await the async scraper
        df = await scraper.scrape_urls(urls)

        # Scraping statistics
        stats = scraper.get_stats()
        logger.info(f"Scraping stats: {stats}")

        # Error handling
        if scraper.errors:
            error_summary = scraper.get_error_summary()
            logger.warning(f"Errors: {error_summary['total_errors']}")
            logger.warning(f"By type: {error_summary['by_type']}")
            scraper.export_errors(output_dir / "scraping_errors.json")

        # Export data if we actually scraped something
        if not df.empty:
            logger.info(f"Extracted {len(df)} board member records")

            # Excel with summary & error sheet
            scraper.export_to_excel(
                df,
                output_dir / "board_members_1394_1404.xlsx",
                include_summary=True,
                include_errors=True,
            )

            # CSV
            scraper.export_to_csv(
                df,
                output_dir / "board_members_1394_1404.csv",
            )

            # Network graph
            if scraper.network.number_of_nodes() > 0:
                scraper.visualize_network(
                    output_dir / "board_network_1394_1404.html"
                )
        else:
            logger.warning("Scraper returned an empty DataFrame.")

    # ====== Step 3: Analyze the Data ======
    if df.empty:
        logger.warning("No board member records extracted – skipping analysis.")
        logger.info(f"All output (URLs, errors) saved to: {output_dir.absolute()}")
        return

    logger.info("Analyzing board member data...")

    # Use DataProcessor for a richer summary (normalizes text, dates, etc.)
    processor = DataProcessor(df)
    processed_df = processor.df  # already normalized

    summary = processor.get_summary_stats()

    print("\n" + "=" * 50)
    print("Board Member Analysis")
    print("=" * 50)
    print(f"Total records: {summary['total_records']}")
    print(f"Columns: {len(summary['columns'])}")
    print(f"Memory usage: {summary['memory_usage_mb']} MB")

    if "company" in processed_df.columns:
        print(f"Unique companies: {processed_df['company'].nunique()}")

    if "new_member" in processed_df.columns:
        print(f"Unique board members: {processed_df['new_member'].nunique()}")

    if "position" in processed_df.columns:
        print("\nPositions distribution:")
        print(processed_df["position"].value_counts().head(10))

    if "degree" in processed_df.columns:
        print("\nEducation levels:")
        print(processed_df["degree"].value_counts().head(10))

    if "is_independent" in processed_df.columns:
        print(f"\nIndependent members: {processed_df['is_independent'].sum()}")

    print("=" * 50 + "\n")

    logger.info(f"All output saved to: {output_dir.absolute()}")


async def example_from_csv():
    """
    Example: Scrape from a CSV file of URLs.
    Uses the URL CSV generated in main().
    """
    logger.info("Scraping from CSV file of URLs...")

    csv_path = Path("output/letters_board_changes_1394_1404.csv")

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    async with BoardMemberScraper() as scraper:
        df = await scraper.scrape_from_csv(str(csv_path), url_column="url")

        if not df.empty:
            scraper.export_to_excel(
                df,
                "output/board_members_from_csv.xlsx",
                include_summary=True,
                include_errors=True,
            )
            logger.info(f"Scraped {len(df)} records from CSV URLs")
        else:
            logger.warning("No records scraped from CSV URLs.")


if __name__ == "__main__":
    asyncio.run(main())
    # If you want to test the CSV-based flow afterwards, uncomment:
    # asyncio.run(example_from_csv())
