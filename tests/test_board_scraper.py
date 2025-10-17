"""
Test script for board member scraper functionality
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from codal_scraper import CodalClient, BoardMemberScraper
from codal_scraper.constants import YEAR_RANGES


async def test_board_scraper():
    """
    Test the board scraper with a small set of URLs
    """
    print("=" * 70)
    print("TESTING BOARD MEMBER SCRAPER")
    print("=" * 70)
    
    # Step 1: Get some board change URLs
    print("\n1. Fetching board change URLs from Codal API...")
    client = CodalClient()
    
    # Get just a few board changes for testing
    board_changes = (client
                    .set_letter_code('ن-45')
                    .set_date_range("1403/01/01", "1403/01/10")
                    .fetch_all_pages(max_pages=1))
    
    if not board_changes:
        print("   [FAIL] No board changes found")
        return
    
    print(f"   [OK] Found {len(board_changes)} board changes")
    
    # Step 2: Extract URLs (just take first 2 for testing)
    urls = client.extract_letter_urls(board_changes[:2])
    print(f"\n2. Testing with {len(urls)} URLs:")
    for i, url in enumerate(urls, 1):
        print(f"   {i}. {url[:80]}...")
    
    if not urls:
        print("   [FAIL] No URLs extracted")
        return
    
    # Step 3: Test the board scraper
    print("\n3. Initializing board scraper...")
    try:
        scraper = BoardMemberScraper()
        print("   [OK] BoardMemberScraper initialized")
    except Exception as e:
        print(f"   [FAIL] Failed to initialize scraper: {e}")
        return
    
    # Step 4: Scrape the URLs
    print("\n4. Scraping board member data...")
    print("   (This may take a few seconds...)")
    
    try:
        board_members_df = await scraper.scrape_urls(urls)
        
        if board_members_df.empty:
            print("   [WARNING] No data scraped (pages might not contain board info)")
        else:
            print(f"   [OK] Scraped {len(board_members_df)} board member records")
            print(f"   [OK] Columns: {board_members_df.columns.tolist()[:5]}...")
            
            # Show some stats
            if 'company' in board_members_df.columns:
                print(f"   [OK] Companies: {board_members_df['company'].nunique()}")
            if 'new_member' in board_members_df.columns:
                print(f"   [OK] Board members: {board_members_df['new_member'].nunique()}")
            
            # Export to Excel
            print("\n5. Exporting data...")
            scraper.export_to_excel(board_members_df, 'test_board_output.xlsx')
            print("   [OK] Exported to test_board_output.xlsx")
            
    except Exception as e:
        print(f"   [FAIL] Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("BOARD SCRAPER TEST COMPLETE!")
    print("=" * 70)
    
    return board_members_df


async def test_csv_workflow():
    """
    Test the CSV-based workflow (like the original mycode.py)
    """
    print("\n" + "=" * 70)
    print("TESTING CSV WORKFLOW")
    print("=" * 70)
    
    # Step 1: Create a CSV file with URLs
    print("\n1. Creating data_temp.csv...")
    client = CodalClient()
    
    board_changes = (client
                    .set_letter_code('ن-45')
                    .set_date_range("1403/01/01", "1403/01/05")
                    .fetch_all_pages(max_pages=1))
    
    if board_changes:
        client.save_urls_to_csv(board_changes[:3], 'data_temp.csv')
        print(f"   [OK] Saved {min(3, len(board_changes))} URLs to data_temp.csv")
        
        # Step 2: Scrape from CSV
        print("\n2. Scraping from CSV file...")
        scraper = BoardMemberScraper()
        
        try:
            df = await scraper.scrape_from_csv('data_temp.csv')
            
            if df.empty:
                print("   [WARNING] No data scraped")
            else:
                print(f"   [OK] Scraped {len(df)} records from CSV")
                scraper.export_to_excel(df, 'csv_workflow_output.xlsx')
                print("   [OK] Exported to csv_workflow_output.xlsx")
                
        except Exception as e:
            print(f"   [FAIL] CSV scraping failed: {e}")
    else:
        print("   [FAIL] No board changes found")
    
    print("\n" + "=" * 70)
    print("CSV WORKFLOW TEST COMPLETE!")
    print("=" * 70)


def main():
    """
    Run the tests
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Board Member Scraper')
    parser.add_argument('--csv', action='store_true', help='Test CSV workflow')
    parser.add_argument('--both', action='store_true', help='Run both tests')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("BOARD MEMBER SCRAPER TEST")
    print("=" * 70)
    print("\nNote: This test requires crawlee[playwright] to be installed.")
    print("Install with: pip install crawlee[playwright] && playwright install chromium")
    
    if args.csv:
        asyncio.run(test_csv_workflow())
    elif args.both:
        asyncio.run(test_board_scraper())
        asyncio.run(test_csv_workflow())
    else:
        asyncio.run(test_board_scraper())


if __name__ == "__main__":
    main()