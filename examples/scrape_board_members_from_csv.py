#!/usr/bin/env python
import asyncio
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

from codal_scraper import BoardMemberScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("board-members-from-csv")


# ====== CONFIG ======

# CSV produced by CodalClient.save_urls_to_csv(...)
INPUT_CSV = Path("output/letters_board_changes_1394_1404.csv")  # adjust if needed

# Where we store scraped board-member rows (incrementally appended)
OUTPUT_CSV = Path("output/board_members_1394_1404_from_urls.csv")

# Accumulated scraping errors (JSON list of error dicts)
ERRORS_JSON = Path("output/board_members_1394_1404_errors.json")

# Must be <= 1000 because BoardMemberScraper internally uses
# PlaywrightCrawler(max_requests_per_crawl=1000)
CHUNK_SIZE = 800  # 15_083 URLs ≈ 20 chunks of 800

# Scraper tuning
MAX_RETRIES = 2
TIMEOUT = 60          # seconds
MAX_CONCURRENT = 5    # stored on scraper (Crawlee handles concurrency internally)


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    """Split a list into evenly-sized chunks."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def scrape_chunk(
    chunk_urls: List[str],
    chunk_index: int,
    total_chunks: int,
    output_csv: Path,
    errors_json: Path,
) -> int:
    """
    Scrape a single chunk of URLs and append results to OUTPUT_CSV.

    Returns:
        Number of rows written for this chunk.
    """
    logger.info(
        f"[Chunk {chunk_index}/{total_chunks}] Scraping {len(chunk_urls)} URLs..."
    )

    # Fresh scraper per chunk => bounded memory, clean network graph
    scraper = BoardMemberScraper(
        max_retries=MAX_RETRIES,
        timeout=TIMEOUT,
        headless=True,
        max_concurrent=MAX_CONCURRENT,
    )

    df = await scraper.scrape_urls(chunk_urls)

    if df.empty:
        logger.warning(f"[Chunk {chunk_index}] No data extracted.")
        return 0

    # Append to CSV incrementally to keep memory usage low
    header = not output_csv.exists()
    df.to_csv(
        output_csv,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )
    logger.info(
        f"[Chunk {chunk_index}] Wrote {len(df)} rows to {output_csv.name}"
    )

    # Merge errors into a single JSON list of dicts
    if scraper.errors:
        existing_errors = []
        if errors_json.exists():
            try:
                existing_errors = json.loads(
                    errors_json.read_text(encoding="utf-8")
                )
            except Exception as e:
                logger.warning(
                    f"Could not read existing errors JSON ({e}), overwriting."
                )

        existing_errors.extend([e.to_dict() for e in scraper.errors])
        errors_json.write_text(
            json.dumps(existing_errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.warning(
            f"[Chunk {chunk_index}] Collected {len(scraper.errors)} errors "
            f"(total now {len(existing_errors)})."
        )

    # Help GC
    del scraper
    del df

    return len(chunk_urls)


async def main():
    # ====== Step 1: Load & clean URL list ======
    if not INPUT_CSV.exists():
        logger.error(f"Input CSV not found: {INPUT_CSV}")
        return

    logger.info(f"Loading URLs from {INPUT_CSV} ...")

    # Only the URL column to keep memory under control
    try:
        df_urls = pd.read_csv(INPUT_CSV, usecols=["url"])
    except ValueError:
        # Fallback: read all columns, then pick what's available
        df_all = pd.read_csv(INPUT_CSV)
        if "url" not in df_all.columns:
            logger.error(f"'url' column not found in {INPUT_CSV}")
            return
        df_urls = df_all[["url"]]

    # Basic cleaning
    df_urls = df_urls[df_urls["url"].notna()]
    df_urls["url"] = df_urls["url"].astype(str).str.strip()
    df_urls = df_urls[df_urls["url"].str.startswith("http")]

    # Deduplicate
    df_urls = df_urls.drop_duplicates(subset=["url"])
    logger.info(f"Total unique URLs in CSV: {len(df_urls)}")

    # ====== Step 2: Resume support – skip already scraped URLs ======
    if OUTPUT_CSV.exists():
        logger.info(f"Existing output file found: {OUTPUT_CSV}")
        try:
            already_df = pd.read_csv(OUTPUT_CSV, usecols=["url"])
            scraped_set = set(
                already_df["url"].dropna().astype(str).str.strip().tolist()
            )
            logger.info(f"Already scraped URLs: {len(scraped_set)}")

            df_urls = df_urls[~df_urls["url"].isin(scraped_set)]
            logger.info(f"Remaining URLs to scrape: {len(df_urls)}")
        except Exception as e:
            logger.warning(
                f"Could not read existing output for resume logic: {e}. "
                "Proceeding to scrape all URLs."
            )

    if df_urls.empty:
        logger.info("No new URLs to scrape. Nothing to do.")
        return

    urls = df_urls["url"].tolist()
    logger.info(f"Preparing to scrape {len(urls)} URLs...")

    chunks = chunk_list(urls, CHUNK_SIZE)
    total_chunks = len(chunks)

    logger.info(
        f"Prepared {total_chunks} chunks of up to {CHUNK_SIZE} URLs each "
        f"(safe under max_requests_per_crawl=1000)."
    )

    # ====== Step 3: Scrape chunks sequentially ======
    # Each chunk uses Crawlee+Playwright internally with concurrency.
    total_urls_attempted = 0
    for idx, chunk_urls in enumerate(chunks, start=1):
        try:
            urls_in_chunk = len(chunk_urls)
            total_urls_attempted += urls_in_chunk

            rows_written = await scrape_chunk(
                chunk_urls=chunk_urls,
                chunk_index=idx,
                total_chunks=total_chunks,
                output_csv=OUTPUT_CSV,
                errors_json=ERRORS_JSON,
            )

            logger.info(
                f"[Chunk {idx}] Done. URLs in chunk: {urls_in_chunk}, "
                f"rows written: {rows_written}."
            )
        except Exception as e:
            logger.exception(f"Fatal error in chunk {idx}: {e}")

    logger.info(
        f"All chunks finished. Total URLs attempted: {total_urls_attempted}."
    )
    logger.info(f"Board member data CSV: {OUTPUT_CSV.absolute()}")
    if ERRORS_JSON.exists():
        logger.info(f"Error log JSON: {ERRORS_JSON.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
