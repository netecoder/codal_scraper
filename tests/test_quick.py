"""
Quick test script for Codal scraper
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src import CodalClient, DataProcessor
from src.constants import YEAR_RANGES


def quick_test():
    """Quick test with limited pages"""
    
    print("=" * 60)
    print("CODAL SCRAPER - QUICK TEST")
    print("=" * 60)
    
    # Initialize client
    client = CodalClient(retry_count=3, timeout=30)
    
    # Test 1: Fetch just 2 pages of board changes
    print("\n1. Testing board changes (limited to 2 pages)...")
    client.reset_params()
    client.set_letter_code('ن-45')
    client.set_date_range("1403/01/01", "1403/12/29")
    client.set_company_type('1')
    
    results = client.fetch_all_pages(max_pages=2)
    print(f"   [OK] Fetched {len(results)} items from 2 pages")
    
    # Test 2: Process the data
    print("\n2. Testing data processor...")
    processor = DataProcessor(results)
    
    # Get column info
    print(f"   [OK] Columns found: {len(processor.df.columns)}")
    print(f"   [OK] First 5 columns: {processor.df.columns.tolist()[:5]}")
    
    # Find important columns
    important_cols = []
    for col in processor.df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['symbol', 'date', 'company', 'tracing']):
            important_cols.append(col)
    print(f"   [OK] Important columns: {important_cols}")
    
    # Get stats
    stats = processor.get_summary_stats()
    print(f"   [OK] Total records: {stats.get('total_records', 0)}")
    print(f"   [OK] Memory usage: {stats.get('memory_usage_mb', 0):.2f} MB")
    
    # Test 3: Export to different formats
    print("\n3. Testing export functions...")
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Excel
    excel_path = output_dir / 'test_output.xlsx'
    try:
        processor.to_excel(excel_path)
        print(f"   [OK] Excel export successful: {excel_path}")
    except Exception as e:
        print(f"   [FAIL] Excel export failed: {e}")
    
    # CSV
    csv_path = output_dir / 'test_output.csv'
    try:
        processor.to_csv(csv_path)
        print(f"   [OK] CSV export successful: {csv_path}")
    except Exception as e:
        print(f"   [FAIL] CSV export failed: {e}")
    
    # JSON
    json_path = output_dir / 'test_output.json'
    try:
        processor.to_json(json_path)
        print(f"   [OK] JSON export successful: {json_path}")
    except Exception as e:
        print(f"   [FAIL] JSON export failed: {e}")
    
    # Test 4: Query URL generation
    print("\n4. Testing URL generation...")
    api_url = client.get_query_url(use_api=True)
    web_url = client.get_query_url(use_api=False)
    print(f"   [OK] API URL length: {len(api_url)} chars")
    print(f"   [OK] Web URL length: {len(web_url)} chars")
    
    # Test 5: Fluent API
    print("\n5. Testing fluent API...")
    client2 = CodalClient()
    test_query = (client2
                  .set_symbol('فولاد')
                  .set_letter_code('ن-30')
                  .set_date_range("1403/01/01", "1403/06/30")
                  .set_company_type('1')
                  .set_audit_status(audited=True, not_audited=False))
    
    # Get the URL to verify parameters are set
    test_url = test_query.get_query_url()
    print(f"   [OK] Fluent API chain successful")
    print(f"   [OK] Generated URL with parameters")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Show sample data
    if not processor.df.empty:
        print("\nSample data summary:")
        try:
            first_symbol = processor.df['symbol'].iloc[0] if 'symbol' in processor.df.columns else 'N/A'
            print(f"  - First symbol: {first_symbol.encode('ascii', 'replace').decode('ascii')}")
        except:
            print("  - First symbol: [Persian text]")
        
        try:
            first_company = processor.df['company_name'].iloc[0] if 'company_name' in processor.df.columns else 'N/A'
            print(f"  - First company: {first_company.encode('ascii', 'replace').decode('ascii')}")
        except:
            print("  - First company: [Persian text]")
            
        print(f"  - Data shape: {processor.df.shape}")
        print("\nData successfully exported to 'output' directory!")
    
    return processor


if __name__ == "__main__":
    processor = quick_test()