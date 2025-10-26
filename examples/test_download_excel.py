"""
Quick Test: Download Excel files from financial statements
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from codal_scraper import CodalClient


def safe_print(message, fallback=None):
    """Print with fallback for Unicode encoding errors"""
    try:
        print(message)
    except UnicodeEncodeError:
        if fallback:
            print(fallback)
        else:
            # Remove problematic characters
            safe_message = message.encode('ascii', 'ignore').decode('ascii')
            print(safe_message)


def test_download_financial_excel():
    """
    Quick test of downloading Excel files from financial statements
    """
    
    # Initialize client
    client = CodalClient(retry_count=3, timeout=30)
    
    print("=" * 60)
    print("TESTING EXCEL FILE DOWNLOADER")
    print("=" * 60)
    
    # Use a very short date range for quick testing
    from_date = "1403/01/01"
    to_date = "1403/01/29"
    
    print(f"\nTest Parameters:")
    print(f"  Date Range: {from_date} to {to_date}")
    print(f"  Output Directory: test_financial_excel/")
    print(f"  Max Files: 10 (for quick testing)")
    print()
    
    try:
        # Download Excel files
        downloaded_files = client.download_financial_excel_files(
            from_date=from_date,
            to_date=to_date,
            period_length=12,
            audited_only=True,
            output_dir="test_financial_excel",
            max_files=10
        )
        
        print(f"\n{'=' * 60}")
        print(f"Test Results:")
        print(f"  Total files downloaded: {len(downloaded_files)}")
        print(f"{'=' * 60}\n")
        
        if downloaded_files:
            print("Successfully downloaded files:")
            for i, file_path in enumerate(downloaded_files, 1):
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                # Use safe print to avoid Unicode errors
                safe_print(
                    f"  {i}. {file_path} ({file_size:,} bytes)",
                    f"  {i}. File {i} ({file_size:,} bytes)"
                )
        else:
            print("No Excel files were downloaded.")
            print("\nNote: Financial statements may not always include Excel attachments.")
        
        return downloaded_files
        
    except Exception as e:
        safe_print(
            f"\nError during download: {e}",
            f"\nError during download (see logs above)"
        )
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    files = test_download_financial_excel()
    
    if files:
        safe_print(
            f"\nTest completed! Downloaded {len(files)} Excel files to 'test_financial_excel/'",
            f"\nTest completed! Downloaded {len(files)} Excel files."
        )
    else:
        print("\nNo files were downloaded. Financial statements may not include Excel attachments.")
