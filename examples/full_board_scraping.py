"""
Full Board Member Scraping Workflow
This example demonstrates the complete workflow:
1. Fetch board change announcements from Codal API
2. Extract URLs from the results
3. Scrape detailed board member information from each URL
4. Export and visualize the results
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import networkx as nx
from src import CodalClient, DataProcessor, BoardMemberScraper
from src.constants import YEAR_RANGES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def full_board_scraping_workflow():
    """
    Complete workflow for board member data extraction and analysis
    """
    
    print("=" * 70)
    print("CODAL BOARD MEMBER SCRAPING - FULL WORKFLOW")
    print("=" * 70)
    
    # Step 1: Fetch board change announcements from Codal API
    print("\n📋 Step 1: Fetching board change announcements from Codal API...")
    print("-" * 50)
    
    client = CodalClient(retry_count=3, timeout=30)
    
    # Search for board changes in recent years
    # You can adjust the date range as needed
    start_date = YEAR_RANGES[1402][0]  # "1402/01/01"
    end_date = YEAR_RANGES[1403][1]    # "1403/12/29"
    
    # Fetch board changes (limiting to 5 pages for demo)
    board_changes = (client
                    .set_letter_code('ن-45')  # Board changes letter code
                    .set_date_range(start_date, end_date)
                    .set_company_type('1')  # Main stock exchange only
                    .set_entity_type(include_childs=False, include_mains=True)
                    .fetch_all_pages(max_pages=5))  # Limit for demo
    
    print(f"✅ Found {len(board_changes)} board change announcements")
    
    if not board_changes:
        print("❌ No board changes found. Exiting.")
        return
    
    # Step 2: Save URLs to CSV for processing
    print("\n📋 Step 2: Extracting and saving URLs...")
    print("-" * 50)
    
    # Save URLs with metadata
    csv_path = 'data_temp.csv'
    client.save_urls_to_csv(board_changes, csv_path)
    
    # Also save the raw API data
    processor = DataProcessor(board_changes)
    processor.to_excel('api_board_changes.xlsx')
    print(f"✅ Saved {len(board_changes)} URLs to {csv_path}")
    print(f"✅ Saved API data to api_board_changes.xlsx")
    
    # Get statistics from API data
    stats = processor.get_summary_stats()
    print(f"\n📊 API Data Statistics:")
    print(f"   - Unique companies: {stats.get('unique_symbols', 0)}")
    print(f"   - Date range: {stats.get('date_range', {})}")
    
    # Step 3: Scrape detailed board member information
    print("\n📋 Step 3: Scraping detailed board member information...")
    print("-" * 50)
    print("⚠️  This may take several minutes depending on the number of URLs...")
    
    # Initialize the board scraper
    scraper = BoardMemberScraper()
    
    # Scrape board member data from the URLs
    board_members_df = await scraper.scrape_from_csv(csv_path)
    
    if board_members_df.empty:
        print("❌ No board member data scraped. Check the logs for errors.")
        return
    
    print(f"✅ Scraped {len(board_members_df)} board member records")
    print(f"   - Unique companies: {board_members_df['company'].nunique()}")
    print(f"   - Unique board members: {board_members_df['new_member'].nunique()}")
    
    # Step 4: Export the scraped data
    print("\n📋 Step 4: Exporting scraped data...")
    print("-" * 50)
    
    # Export to Excel with summary
    excel_path = 'board_members_detailed.xlsx'
    scraper.export_to_excel(board_members_df, excel_path, include_summary=True)
    print(f"✅ Exported to Excel: {excel_path}")
    
    # Export to CSV for further analysis
    csv_output = 'board_members_detailed.csv'
    scraper.export_to_csv(board_members_df, csv_output)
    print(f"✅ Exported to CSV: {csv_output}")
    
    # Step 5: Create network visualization
    print("\n📋 Step 5: Creating network visualization...")
    print("-" * 50)
    
    if scraper.network and scraper.network.number_of_nodes() > 0:
        network_file = 'board_network.html'
        scraper.visualize_network(network_file)
        print(f"✅ Network visualization saved to: {network_file}")
        print(f"   - Total nodes: {scraper.network.number_of_nodes()}")
        print(f"   - Total edges: {scraper.network.number_of_edges()}")
    else:
        print("⚠️  No network data available for visualization")
    
    # Step 6: Analyze the results
    print("\n📋 Step 6: Analyzing results...")
    print("-" * 50)
    
    # Analyze board composition
    if 'is_independent' in board_members_df.columns:
        independent_pct = (board_members_df['is_independent'].sum() / len(board_members_df)) * 100
        print(f"   - Independent members: {independent_pct:.1f}%")
    
    if 'has_multiple_executive' in board_members_df.columns:
        multi_exec = board_members_df['has_multiple_executive'].sum()
        print(f"   - Members with multiple executive roles: {multi_exec}")
    
    # Find most connected board members (if network data exists)
    if scraper.network and scraper.network.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(scraper.network)
        top_members = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\n🏆 Top 5 Most Connected Entities:")
        for i, (entity, centrality) in enumerate(top_members, 1):
            connections = scraper.network.degree(entity)
            print(f"   {i}. {entity[:50]}... ({connections} connections)")
    
    # Print failed URLs if any
    if scraper.failed_urls:
        print(f"\n⚠️  Failed to scrape {len(scraper.failed_urls)} URLs")
        print("   Failed URLs saved to: failed_urls.txt")
        with open('failed_urls.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(scraper.failed_urls))
    
    print("\n" + "=" * 70)
    print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return board_members_df


async def quick_test():
    """
    Quick test with limited data for verification
    """
    print("=" * 70)
    print("QUICK TEST - LIMITED DATA")
    print("=" * 70)
    
    # Step 1: Get just 1 page of data
    client = CodalClient()
    board_changes = (client
                    .set_letter_code('ن-45')
                    .set_date_range("1403/01/01", "1403/06/30")
                    .fetch_all_pages(max_pages=1))
    
    print(f"Found {len(board_changes)} announcements")
    
    if board_changes:
        # Extract first 3 URLs only
        urls = client.extract_letter_urls(board_changes[:3])
        print(f"Testing with {len(urls)} URLs")
        
        # Scrape the URLs
        scraper = BoardMemberScraper()
        df = await scraper.scrape_urls(urls)
        
        if not df.empty:
            print(f"✅ Successfully scraped {len(df)} board member records")
            scraper.export_to_excel(df, 'test_board_members.xlsx')
            print("✅ Test data saved to test_board_members.xlsx")
        else:
            print("❌ No data scraped")
    else:
        print("❌ No board changes found")


def main():
    """
    Main entry point - choose between full workflow or quick test
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Board Member Scraping Workflow')
    parser.add_argument('--test', action='store_true', help='Run quick test with limited data')
    parser.add_argument('--pages', type=int, default=5, help='Maximum pages to fetch (default: 5)')
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(quick_test())
    else:
        asyncio.run(full_board_scraping_workflow())


if __name__ == "__main__":
    # Run the full workflow
    # You can also run the quick test by uncommenting the line below
    asyncio.run(full_board_scraping_workflow())
    # asyncio.run(quick_test())