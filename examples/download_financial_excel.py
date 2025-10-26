"""
Example: Download Excel files from financial statements
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from codal_scraper import CodalClient
from codal_scraper.constants import YEAR_RANGES


def download_financial_excel_example():
    """
    Example of downloading Excel files from financial statements
    """
    
    # Initialize client
    client = CodalClient(retry_count=3, timeout=30)
    
    print("=" * 60)
    print("FINANCIAL STATEMENT EXCEL DOWNLOADER")
    print("=" * 60)
    
    # Define date range for financial statements
    start_date = YEAR_RANGES[1402][0]  # "1402/01/01"
    end_date = YEAR_RANGES[1402][1]    # "1402/12/29"
    
    print(f"\nDownloading Excel files from financial statements...")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Output Directory: financial_excel/")
    
    # Download Excel files
    # Limit to first 10 files for demo purposes
    downloaded_files = client.download_financial_excel_files(
        from_date=start_date,
        to_date=end_date,
        period_length=12,  # Annual reports
        audited_only=True,
        output_dir="financial_excel",
        max_files=10  # Limit to 10 files for demo
    )
    
    print(f"\n{'=' * 60}")
    print(f"Download Summary:")
    print(f"  Total files downloaded: {len(downloaded_files)}")
    print(f"{'=' * 60}\n")
    
    # Display downloaded files
    if downloaded_files:
        print("Downloaded files:")
        for i, file_path in enumerate(downloaded_files, 1):
            print(f"  {i}. {file_path}")
    
    return downloaded_files


if __name__ == "__main__":
    try:
        # Download Excel files from financial statements
        files = download_financial_excel_example()
        
        print("\n" + "=" * 60)
        print("Download completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
