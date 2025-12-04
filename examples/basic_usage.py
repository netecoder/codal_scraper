#!/usr/bin/env python
"""
Basic usage example for Codal Scraper

This example demonstrates how to use the CodalClient to fetch
announcements from Codal.ir and process them with DataProcessor.
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating basic usage"""
    
    # Import the library
    from codal_scraper import CodalClient, DataProcessor
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # ============== Example 1: Search Board Changes ==============
    logger.info("Example 1: Searching for board changes...")
    
    with CodalClient() as client:
        # Search for board of directors changes in 1402
        letters = client.search_board_changes(
            from_date="1402/01/01",
            to_date="1402/01/30",
            max_pages=5  # Limit for demo
        )
        
        logger.info(f"Found {len(letters)} board change announcements")
        
        # Get query statistics
        stats = client.get_summary_stats()
        logger.info(f"Query stats: {stats}")
    
    # Process the data
    if letters:
        processor = DataProcessor(letters)
        
        # Add letter descriptions
        processor.add_letter_descriptions()
        
        # Show summary
        processor.print_info()
        
        # Export to Excel
        processor.to_excel(output_dir / "board_changes.xlsx")
        
        # Export to CSV
        processor.to_csv(output_dir / "board_changes.csv")
    
    # ============== Example 2: Search by Symbol ==============
    logger.info("\nExample 2: Searching by symbol...")
    
    with CodalClient() as client:
        # Search all announcements for a specific symbol
        letters = client.search_by_symbol(
            symbol="فولاد",
            from_date="1402/01/01",
            to_date="1402/01/30",
            max_pages=3
        )
        
        logger.info(f"Found {len(letters)} announcements for فولاد")
    
    if letters:
        processor = DataProcessor(letters)
        processor.add_letter_descriptions()
        
        # Filter and sort
        processor.sort_by('publish_date_time', ascending=False)
        
        # Show first few records
        print("\nFirst 5 announcements:")
        print(processor.head())
        
        # Export
        processor.to_excel(output_dir / "foolad_announcements.xlsx")
    
    # ============== Example 3: Financial Statements ==============
    logger.info("\nExample 3: Searching for financial statements...")
    
    with CodalClient() as client:
        # Search for annual financial statements
        letters = client.search_financial_statements(
            from_date="1402/01/01",
            to_date="1402/01/30",
            period_length=12,  # Annual
            audited_only=True,
            max_pages=5
        )
        
        logger.info(f"Found {len(letters)} financial statements")
    
    if letters:
        processor = DataProcessor(letters)
        processor.add_letter_descriptions()
        
        # Get summary statistics
        summary = processor.get_summary_stats()
        logger.info(f"Unique symbols: {summary.get('unique_symbols', 0)}")
        
        # Aggregate by symbol
        aggregated = processor.aggregate_by_symbol_year()
        print("\nAnnouncements per symbol per year:")
        print(aggregated.head(10))
        
        # Export
        processor.to_excel(output_dir / "financial_statements.xlsx")
    
    # ============== Example 4: Method Chaining ==============
    logger.info("\nExample 4: Using method chaining...")
    
    with CodalClient() as client:
        letters = client.search_board_changes(
            from_date="1402/01/01",
            to_date="1402/01/30",
            max_pages=10
        )
    
    if letters:
        # Chain multiple operations
        (DataProcessor(letters)
            .add_letter_descriptions()
            .filter_by_date_range("1402/01/01", "1402/01/30")
            .remove_duplicates(subset=['symbol', 'tracing_no'])
            .sort_by('publish_date_time', ascending=False)
            .fill_na('')
            .to_excel(output_dir / "filtered_board_changes.xlsx")
            .to_csv(output_dir / "filtered_board_changes.csv"))
        
        logger.info("Exported filtered data")
    
    # ============== Example 5: Custom Filters ==============
    logger.info("\nExample 5: Using custom filters...")
    
    with CodalClient() as client:
        letters = client.search_financial_statements(
            from_date="1402/01/01",
            to_date="1402/01/30",
            max_pages=5
        )
    
    if letters:
        processor = DataProcessor(letters)
        
        # Filter by custom condition
        processor.filter_by_condition(
            lambda df: df['has_excel'] == True if 'has_excel' in df.columns else True,
            description="has_excel=True"
        )
        
        # Get transformations applied
        print("\nTransformations applied:")
        for t in processor.get_transformations():
            print(f"  - {t}")
        
        processor.to_json(output_dir / "statements_with_excel.json")
    
    logger.info("\nAll examples completed!")
    logger.info(f"Output files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()