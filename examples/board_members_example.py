#!/usr/bin/env python
"""
Board members scraping example for Codal Scraper

This example demonstrates how to scrape detailed board member
information from Codal announcement pages.
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main function demonstrating board member scraping"""
    
    from codal_scraper import CodalClient, BoardMemberScraper, DataProcessor
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # ============== Step 1: Fetch Board Change Announcements ==============
    logger.info("Step 1: Fetching board change announcements...")
    
    with CodalClient() as client:
        # Search for board changes
        letters = client.search_board_changes(
            from_date="1402/06/01",
            to_date="1402/12/29",
            max_pages=5  # Limit for demo
        )
        
        logger.info(f"Found {len(letters)} announcements")
        
        # Extract URLs
        urls = client.extract_letter_urls(letters)
        logger.info(f"Extracted {len(urls)} URLs")
        
        # Save URLs to CSV for reference
        client.save_urls_to_csv(letters, output_dir / "letter_urls.csv")
    
    if not urls:
        logger.warning("No URLs to scrape")
        return
    
    # Limit URLs for demo
    urls = urls[:10]
    logger.info(f"Scraping first {len(urls)} URLs for demo...")
    
    # ============== Step 2: Scrape Board Member Details ==============
    logger.info("\nStep 2: Scraping board member details...")
    
    async with BoardMemberScraper(
        max_retries=2,
        timeout=60,
        headless=True,
        max_concurrent=3
    ) as scraper:
        # Scrape the URLs
        df = await scraper.scrape_urls(urls)
        
        # Get statistics
        stats = scraper.get_stats()
        logger.info(f"Scraping stats: {stats}")
        
        # Check for errors
        if scraper.errors:
            error_summary = scraper.get_error_summary()
            logger.warning(f"Errors: {error_summary['total_errors']}")
            logger.warning(f"By type: {error_summary['by_type']}")
            
            # Export errors
            scraper.export_errors(output_dir / "scraping_errors.json")
        
        # Export data
        if not df.empty:
            logger.info(f"Extracted {len(df)} board member records")
            
            # Export to Excel with summary
            scraper.export_to_excel(
                df,
                output_dir / "board_members.xlsx",
                include_summary=True,
                include_errors=True
            )
            
            # Export to CSV
            scraper.export_to_csv(df, output_dir / "board_members.csv")
            
            # Export network
            if scraper.network.number_of_nodes() > 0:
                scraper.visualize_network(output_dir / "board_network.html")
    
    # ============== Step 3: Analyze the Data ==============
    if not df.empty:
        logger.info("\nStep 3: Analyzing board member data...")
        
        processor = DataProcessor(df)
        
        # Get summary
        summary = processor.get_summary_stats()
        
        print("\n" + "="*50)
        print("Board Member Analysis")
        print("="*50)
        print(f"Total records: {summary['total_records']}")
        
        if 'company' in df.columns:
            print(f"Unique companies: {df['company'].nunique()}")
        
        if 'new_member' in df.columns:
            print(f"Unique board members: {df['new_member'].nunique()}")
        
        if 'position' in df.columns:
            print("\nPositions distribution:")
            print(df['position'].value_counts().head(10))
        
        if 'degree' in df.columns:
            print("\nEducation levels:")
            print(df['degree'].value_counts().head(10))
        
        if 'is_independent' in df.columns:
            print(f"\nIndependent members: {df['is_independent'].sum()}")
        
        print("="*50 + "\n")
    
    logger.info(f"All output saved to: {output_dir.absolute()}")


async def example_from_csv():
    """Example: Scrape from a CSV file of URLs"""
    from codal_scraper import BoardMemberScraper
    
    logger.info("Scraping from CSV file...")
    
    csv_path = Path("output/letter_urls.csv")
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    async with BoardMemberScraper() as scraper:
        df = await scraper.scrape_from_csv(str(csv_path), url_column='url')
        
        if not df.empty:
            scraper.export_to_excel(df, "output/board_members_from_csv.xlsx")
            logger.info(f"Scraped {len(df)} records from CSV URLs")


async def example_network_analysis():
    """Example: Analyze board member network"""
    from codal_scraper import CodalClient, BoardMemberScraper
    import networkx as nx
    
    logger.info("Network analysis example...")
    
    # First, scrape some data
    with CodalClient() as client:
        letters = client.search_board_changes(
            from_date="1402/01/01",
            to_date="1402/03/31",
            max_pages=3
        )
        urls = client.extract_letter_urls(letters)[:5]
    
    if not urls:
        return
    
    async with BoardMemberScraper() as scraper:
        await scraper.scrape_urls(urls)
        
        # Analyze network
        network = scraper.network
        
        if network.number_of_nodes() == 0:
            logger.info("No network data")
            return
        
        print("\n" + "="*50)
        print("Network Analysis")
        print("="*50)
        
        # Basic stats
        print(f"Nodes: {network.number_of_nodes()}")
        print(f"Edges: {network.number_of_edges()}")
        print(f"Density: {nx.density(network):.4f}")
        
        # Node types
        companies = [n for n, d in network.nodes(data=True) if d.get('node_type') == 'company']
        people = [n for n, d in network.nodes(data=True) if d.get('node_type') == 'person']
        print(f"Companies: {len(companies)}")
        print(f"People: {len(people)}")
        
        # Most connected people (serve on multiple boards)
        if people:
            person_degrees = {p: network.degree(p) for p in people}
            top_connected = sorted(person_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print("\nMost connected board members:")
            for person, degree in top_connected:
                print(f"  {person}: {degree} connections")
        
        # Save visualization
        scraper.visualize_network("output/network_analysis.html")
        print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
    
    # Uncomment to run additional examples:
    # asyncio.run(example_from_csv())
    # asyncio.run(example_network_analysis())