"""
Test script for the integrated Codal scraper with board member scraping
This test verifies the integration without requiring async dependencies
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from codal_scraper import CodalClient, DataProcessor
from codal_scraper.constants import YEAR_RANGES

def test_url_extraction():
    """
    Test the URL extraction functionality
    """
    print("=" * 60)
    print("TESTING URL EXTRACTION")
    print("=" * 60)
    
    # Initialize client
    client = CodalClient()
    
    # Fetch a single page of board changes
    print("\n1. Fetching board changes from API...")
    board_changes = (client
                    .set_letter_code('ن-45')
                    .set_date_range("1403/01/01", "1403/01/10")
                    .fetch_all_pages(max_pages=1))
    
    print(f"   [OK] Found {len(board_changes)} board change announcements")
    
    if not board_changes:
        print("   [FAIL] No board changes found")
        return False
    
    # Extract URLs
    print("\n2. Extracting URLs...")
    urls = client.extract_letter_urls(board_changes)
    print(f"   [OK] Extracted {len(urls)} URLs")
    
    # Display first 3 URLs
    print("\n   Sample URLs:")
    for i, url in enumerate(urls[:3], 1):
        print(f"   {i}. {url}")
    
    # Save URLs to CSV
    print("\n3. Saving URLs to CSV...")
    csv_path = 'data_temp.csv'
    client.save_urls_to_csv(board_changes, csv_path)
    print(f"   [OK] Saved URLs to {csv_path}")
    
    # Process API data
    print("\n4. Processing API data...")
    processor = DataProcessor(board_changes)
    
    # Get statistics
    stats = processor.get_summary_stats()
    print(f"   [OK] Total records: {stats.get('total_records', 0)}")
    print(f"   [OK] Unique symbols: {stats.get('unique_symbols', 0)}")
    
    # Export to Excel
    print("\n5. Exporting to Excel...")
    processor.to_excel('test_board_changes.xlsx')
    print(f"   [OK] Exported to test_board_changes.xlsx")
    
    return True


def test_board_scraper_import():
    """
    Test if the board scraper module can be imported
    """
    print("\n" + "=" * 60)
    print("TESTING BOARD SCRAPER MODULE")
    print("=" * 60)
    
    try:
        from codal_scraper.board_scraper import BoardMemberScraper
        print("   [OK] BoardMemberScraper imported successfully")
        
        # Test initialization
        scraper = BoardMemberScraper()
        print("   [OK] BoardMemberScraper initialized")
        
        # Check available methods
        methods = [
            'scrape_urls',
            'scrape_from_csv',
            'export_to_excel',
            'export_to_csv',
            'visualize_network'
        ]
        
        for method in methods:
            if hasattr(scraper, method):
                print(f"   [OK] Method '{method}' available")
            else:
                print(f"   [FAIL] Method '{method}' not found")
        
        return True
        
    except ImportError as e:
        print(f"   [WARNING] Could not import BoardMemberScraper: {e}")
        print("   This is expected if async dependencies are not installed")
        return False


def test_data_flow():
    """
    Test the complete data flow (without async scraping)
    """
    print("\n" + "=" * 60)
    print("TESTING DATA FLOW")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("\n1. Fetching board changes...")
    client = CodalClient()
    board_changes = client.search_board_changes(
        from_date="1403/04/01",
        to_date="1403/06/30",
        company_type='1'
    )[:10]  # Limit to 10 for testing
    
    if not board_changes:
        print("   [FAIL] No data fetched")
        return False
    
    print(f"   [OK] Fetched {len(board_changes)} announcements")
    
    # Step 2: Process data
    print("\n2. Processing data...")
    processor = DataProcessor(board_changes)
    
    # Check columns
    columns = processor.df.columns.tolist()
    important_cols = ['symbol', 'company_name', 'tracing_no']
    
    for col in important_cols:
        if col in columns:
            print(f"   [OK] Column '{col}' found")
        else:
            print(f"   [WARNING] Column '{col}' not found")
    
    # Step 3: Extract URLs and save
    print("\n3. Extracting and saving URLs...")
    urls = client.extract_letter_urls(board_changes)
    
    if urls:
        print(f"   [OK] Extracted {len(urls)} URLs")
        
        # Save to CSV
        client.save_urls_to_csv(board_changes, 'test_urls.csv')
        print("   [OK] Saved URLs to test_urls.csv")
    else:
        print("   [FAIL] No URLs extracted")
    
    # Step 4: Export processed data
    print("\n4. Exporting processed data...")
    processor.to_excel('test_processed_data.xlsx')
    print("   [OK] Exported to test_processed_data.xlsx")
    
    return True


def main():
    """
    Run all tests
    """
    print("\n" + "=" * 70)
    print("CODAL SCRAPER INTEGRATION TEST")
    print("=" * 70)
    
    results = []
    
    # Test 1: URL Extraction
    try:
        result = test_url_extraction()
        results.append(("URL Extraction", result))
    except Exception as e:
        print(f"\n   [ERROR] URL extraction test failed: {e}")
        results.append(("URL Extraction", False))
    
    # Test 2: Board Scraper Import
    try:
        result = test_board_scraper_import()
        results.append(("Board Scraper Module", result))
    except Exception as e:
        print(f"\n   [ERROR] Board scraper test failed: {e}")
        results.append(("Board Scraper Module", False))
    
    # Test 3: Data Flow
    try:
        result = test_data_flow()
        results.append(("Data Flow", result))
    except Exception as e:
        print(f"\n   [ERROR] Data flow test failed: {e}")
        results.append(("Data Flow", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "[OK]" if passed else "[FAIL]"
        print(f"{symbol} {test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\nALL TESTS PASSED! The integration is working correctly.")
    else:
        print("\nSome tests failed. Check the output above for details.")
    
    print("\n" + "=" * 70)
    print("Integration test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()