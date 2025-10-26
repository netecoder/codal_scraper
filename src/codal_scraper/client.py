"""
Main client for interacting with Codal API
"""

import json
import logging
import re
import time
import urllib.parse as urlparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

from .constants import (
    SEARCH_API_URL, REPORT_LIST_URL, DEFAULT_HEADERS,
    PERIOD_LENGTHS, COMPANY_TYPES, CDN_TSETMC_API
)
from .utils import clean_dict, clean_symbol
from .validators import InputValidator, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CodalClient:
    """
    Client for querying Codal.ir API
    
    This client provides a clean interface for searching and fetching
    announcements and reports from the Iranian stock market disclosure system.
    """
    
    def __init__(self, retry_count: int = 3, timeout: int = 30):
        """
        Initialize Codal client
        
        Args:
            retry_count: Number of retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.retry_count = retry_count
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        
        # Query parameters with default values
        self.params = {
            "PageNumber": 1,
            "Symbol": -1,
            "PublisherStatus": -1,
            "Category": -1,
            "CompanyType": -1,
            "CompanyState": -1,
            "LetterType": -1,
            "Subject": -1,
            "TracingNo": -1,
            "LetterCode": -1,
            "Length": -1,
            "FromDate": -1,
            "ToDate": -1,
            "Audited": "true",
            "NotAudited": "true",
            "Consolidatable": "true",
            "NotConsolidatable": "true",
            "Childs": "true",
            "Mains": "true",
            "AuditorRef": -1,
            "YearEndToDate": -1,
            "Publisher": "false"
        }
        
        # Response tracking
        self.total_results = None
        self.total_pages = None
        self.current_page = 1
    
    def reset_params(self) -> None:
        """Reset all query parameters to default values"""
        self.params = {
            "PageNumber": 1,
            "Symbol": -1,
            "PublisherStatus": -1,
            "Category": -1,
            "CompanyType": -1,
            "CompanyState": -1,
            "LetterType": -1,
            "Subject": -1,
            "TracingNo": -1,
            "LetterCode": -1,
            "Length": -1,
            "FromDate": -1,
            "ToDate": -1,
            "Audited": "true",
            "NotAudited": "true",
            "Consolidatable": "true",
            "NotConsolidatable": "true",
            "Childs": "true",
            "Mains": "true",
            "AuditorRef": -1,
            "YearEndToDate": -1,
            "Publisher": "false"
        }
    
    # ============== Parameter Setters ==============
    
    def set_symbol(self, symbol: str) -> 'CodalClient':
        """Set stock symbol for query"""
        if symbol:
            symbol = clean_symbol(symbol)
            InputValidator(symbol).is_symbol()
            self.params['Symbol'] = symbol
        else:
            self.params['Symbol'] = -1
        return self
    
    def set_company_type(self, company_type: str) -> 'CodalClient':
        """
        Set company type
        
        Args:
            company_type: 'بورس', 'فرابورس', 'پایه', or '1', '2', '3'
        """
        if company_type:
            # Convert Persian names to codes
            type_code = COMPANY_TYPES.get(company_type, company_type)
            self.params['CompanyType'] = type_code
        else:
            self.params['CompanyType'] = -1
        return self
    
    def set_isic(self, isic: str) -> 'CodalClient':
        """Set ISIC code"""
        if isic:
            InputValidator(isic).is_isic()
            self.params['Isic'] = isic
        else:
            self.params['Isic'] = -1
        return self
    
    def set_subject(self, subject: str) -> 'CodalClient':
        """Set announcement subject"""
        if subject:
            self.params["Subject"] = subject
        else:
            self.params["Subject"] = -1
        return self
    
    def set_tracing_number(self, tracing_no: str) -> 'CodalClient':
        """Set tracing number"""
        if tracing_no:
            self.params["TracingNo"] = tracing_no
        else:
            self.params["TracingNo"] = -1
        return self
    
    def set_letter_code(self, code: str) -> 'CodalClient':
        """Set letter code (e.g., 'ن-45')"""
        if code:
            self.params["LetterCode"] = code
        else:
            self.params["LetterCode"] = -1
        return self
    
    def set_period_length(self, period: Union[int, str]) -> 'CodalClient':
        """Set reporting period length (1-12 months)"""
        if period:
            period_value = PERIOD_LENGTHS.get(period, -1)
            self.params["Length"] = period_value
        else:
            self.params["Length"] = -1
        return self
    
    def set_date_range(self, from_date: str = None, to_date: str = None) -> 'CodalClient':
        """
        Set date range for query
        
        Args:
            from_date: Start date in YYYY/MM/DD format (Persian calendar)
            to_date: End date in YYYY/MM/DD format (Persian calendar)
        """
        if from_date:
            InputValidator(from_date).is_date()
            self.params["FromDate"] = from_date
        else:
            self.params["FromDate"] = -1
        
        if to_date:
            InputValidator(to_date).is_date()
            self.params["ToDate"] = to_date
        else:
            self.params["ToDate"] = -1
        
        return self
    
    def set_audit_status(self, audited: bool = True, not_audited: bool = True) -> 'CodalClient':
        """Set audit status filters"""
        self.params["Audited"] = str(audited).lower()
        self.params["NotAudited"] = str(not_audited).lower()
        return self
    
    def set_consolidation_status(self, 
                                consolidated: bool = True, 
                                not_consolidated: bool = True) -> 'CodalClient':
        """Set consolidation status filters"""
        self.params["Consolidatable"] = str(consolidated).lower()
        self.params["NotConsolidatable"] = str(not_consolidated).lower()
        return self
    
    def set_entity_type(self, include_childs: bool = True, include_mains: bool = True) -> 'CodalClient':
        """Set entity type filters (subsidiaries and main companies)"""
        self.params["Childs"] = str(include_childs).lower()
        self.params["Mains"] = str(include_mains).lower()
        return self
    
    def set_year_end(self, year_end_date: str) -> 'CodalClient':
        """Set fiscal year end date"""
        if year_end_date:
            InputValidator(year_end_date).is_date()
            self.params["YearEndToDate"] = year_end_date
        else:
            self.params["YearEndToDate"] = -1
        return self
    
    def set_publisher_only(self, publisher_only: bool = False) -> 'CodalClient':
        """Set whether to include only organization-published announcements"""
        self.params["Publisher"] = str(publisher_only).lower()
        return self
    
    def set_page_number(self, page: int) -> 'CodalClient':
        """Set page number for pagination"""
        if page and page > 0:
            self.params["PageNumber"] = page
            self.current_page = page
        return self
    
    # ============== URL Generation ==============
    
    def get_query_url(self, use_api: bool = True) -> str:
        """
        Generate query URL
        
        Args:
            use_api: If True, generate API URL; if False, generate web URL
        
        Returns:
            Complete query URL
        """
        base_url = SEARCH_API_URL if use_api else REPORT_LIST_URL
        cleaned_params = clean_dict(self.params)
        
        url_parts = list(urlparse.urlparse(base_url))
        query = dict(urlparse.parse_qsl(url_parts[4]))
        query.update(cleaned_params)
        
        if use_api:
            url_parts[4] = f"{urlencode(query)}&search=true"
        else:
            url_parts[4] = f"search&{urlencode(query)}"
        
        return urlparse.urlunparse(url_parts)
    
    # ============== Data Fetching ==============
    
    def _make_request(self, url: str) -> Optional[Dict]:
        """
        Make HTTP request with retry logic
        
        Args:
            url: URL to request
        
        Returns:
            Response data as dictionary or None if failed
        """
        for attempt in range(self.retry_count):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Check if response is JSON
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    return response.json()
                else:
                    # Try to parse as JSON anyway
                    return json.loads(response.text)
                    
            except RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All request attempts failed for URL: {url}")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return None
    
    def fetch_page(self, page: int = None) -> Optional[List[Dict]]:
        """
        Fetch a single page of results
        
        Args:
            page: Page number to fetch (if None, uses current page)
        
        Returns:
            List of letters/announcements or None if failed
        """
        if page:
            self.set_page_number(page)
        
        url = self.get_query_url(use_api=True)
        logger.info(f"Fetching page {self.params['PageNumber']}")
        
        response = self._make_request(url)
        
        if not response:
            return None
        
        # Update metadata
        self.total_results = response.get("Total", 0)
        self.total_pages = response.get("Page", 0)
        
        return response.get("Letters", [])
    
    def fetch_all_pages(self, max_pages: int = None) -> List[Dict]:
        """
        Fetch multiple pages of results
        
        Args:
            max_pages: Maximum number of pages to fetch (if None, fetches all)
        
        Returns:
            Combined list of all letters/announcements
        """
        # First, fetch the initial page to get metadata
        first_page_data = self.fetch_page(1)
        
        if not first_page_data:
            logger.warning("Failed to fetch first page")
            return []
        
        all_letters = list(first_page_data)
        
        # Determine how many pages to fetch
        pages_to_fetch = min(self.total_pages, max_pages) if max_pages else self.total_pages
        
        if pages_to_fetch <= 1:
            return all_letters
        
        logger.info(f"Fetching {pages_to_fetch - 1} additional pages (total: {pages_to_fetch})")
        
        # Fetch remaining pages
        for page in range(2, pages_to_fetch + 1):
            page_data = self.fetch_page(page)
            
            if page_data:
                all_letters.extend(page_data)
                logger.info(f"Fetched page {page}/{pages_to_fetch} ({len(page_data)} items)")
            else:
                logger.warning(f"Failed to fetch page {page}")
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(0.5)
        
        logger.info(f"Total items fetched: {len(all_letters)}")
        return all_letters
    
    def fetch_tsetmc_data(self, data_type: str, code: str, date: str) -> Optional[Dict]:
        """
        Fetch data from TSE TMC CDN API
        
        Args:
            data_type: Type of data to fetch
            code: Stock code
            date: Date for the data
        
        Returns:
            Response data or None if failed
        """
        url = CDN_TSETMC_API.format(data=data_type, code=code, date=date)
        return self._make_request(url)
    
    # ============== Convenience Methods ==============
    
    def search_board_changes(self, 
                            from_date: str, 
                            to_date: str,
                            company_type: str = None) -> List[Dict]:
        """
        Search for board of directors changes (letter code ن-45)
        
        Args:
            from_date: Start date
            to_date: End date
            company_type: Optional company type filter
        
        Returns:
            List of board change announcements
        """
        self.reset_params()
        self.set_letter_code('ن-45')
        self.set_date_range(from_date, to_date)
        
        if company_type:
            self.set_company_type(company_type)
        
        self.set_entity_type(include_childs=False, include_mains=True)
        
        return self.fetch_all_pages()
    
    def search_financial_statements(self,
                                   from_date: str,
                                   to_date: str,
                                   period_length: int = 12,
                                   audited_only: bool = True) -> List[Dict]:
        """
        Search for financial statements
        
        Args:
            from_date: Start date
            to_date: End date
            period_length: Period length in months (default: 12 for annual)
            audited_only: If True, only return audited statements
        
        Returns:
            List of financial statement announcements
        """
        self.reset_params()
        self.set_letter_code('ن-10')  # Financial statements
        self.set_date_range(from_date, to_date)
        self.set_period_length(period_length)
        
        if audited_only:
            self.set_audit_status(audited=True, not_audited=False)
        
        return self.fetch_all_pages()
    
    def search_by_symbol(self, 
                        symbol: str, 
                        from_date: str = None, 
                        to_date: str = None) -> List[Dict]:
        """
        Search all announcements for a specific symbol
        
        Args:
            symbol: Stock symbol
            from_date: Optional start date
            to_date: Optional end date
        
        Returns:
            List of announcements for the symbol
        """
        self.reset_params()
        self.set_symbol(symbol)
        
        if from_date and to_date:
            self.set_date_range(from_date, to_date)
        
        return self.fetch_all_pages()
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics from the last query
        
        Returns:
            Dictionary with query statistics
        """
        return {
            'total_results': self.total_results,
            'total_pages': self.total_pages,
            'current_page': self.current_page,
            'query_params': clean_dict(self.params)
        }
    
    def extract_letter_urls(self, letters: List[Dict]) -> List[str]:
        """
        Extract letter URLs from API response
        
        Args:
            letters: List of letter dictionaries from API
        
        Returns:
            List of letter URLs
        """
        urls = []
        base_url = "https://codal.ir"
        
        for letter in letters:
            try:
                # Check for URL field
                if 'Url' in letter and letter['Url']:
                    url = letter['Url']
                    if not url.startswith('http'):
                        url = f"{base_url}/{url.lstrip('/')}"
                    urls.append(url)
                
                # Alternative: construct URL from TracingNo if available
                elif 'TracingNo' in letter and letter['TracingNo']:
                    tracing_no = letter['TracingNo']
                    url = f"{base_url}/Reports/Decision.aspx?LetterSerial={tracing_no}"
                    urls.append(url)
                    
            except Exception as e:
                logger.warning(f"Failed to extract URL from letter: {e}")
                
        return urls
    
    def save_urls_to_csv(self, letters: List[Dict], file_path: str = 'data_temp.csv') -> None:
        """
        Save letter URLs to CSV file for board member scraping
        
        Args:
            letters: List of letter dictionaries from API
            file_path: Output CSV file path
        """
        try:
            urls = self.extract_letter_urls(letters)
            
            # Create DataFrame with URLs and metadata
            data = []
            for letter in letters:
                try:
                    url = None
                    if 'Url' in letter and letter['Url']:
                        url = letter['Url']
                        if not url.startswith('http'):
                            url = f"https://codal.ir/{url.lstrip('/')}"
                    elif 'TracingNo' in letter and letter['TracingNo']:
                        url = f"https://codal.ir/Reports/Decision.aspx?LetterSerial={letter['TracingNo']}"
                    
                    if url:
                        data.append({
                            'url': url,
                            'symbol': letter.get('Symbol', ''),
                            'company_name': letter.get('CompanyName', ''),
                            'tracing_no': letter.get('TracingNo', ''),
                            'letter_code': letter.get('LetterCode', ''),
                            'sent_date': letter.get('SentDateTime', ''),
                            'publish_date': letter.get('PublishDateTime', '')
                        })
                except Exception as e:
                    logger.warning(f"Failed to process letter for CSV: {e}")
            
            if data:
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                logger.info(f"Saved {len(data)} URLs to {file_path}")
            else:
                logger.warning("No URLs to save")
                
        except Exception as e:
            logger.error(f"Failed to save URLs to CSV: {e}")
    
    def _download_file(self, url: str, file_path: Path) -> bool:
        """
        Download a file from URL
        
        Args:
            url: File URL
            file_path: Local file path to save
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Save file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def _extract_excel_links(self, html_content: str, base_url: str = "https://codal.ir") -> List[Dict]:
        """
        Extract Excel file links from HTML content
        
        Args:
            html_content: HTML content from announcement page
            base_url: Base URL for resolving relative links
        
        Returns:
            List of dictionaries with file info (url, filename)
        """
        excel_links = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all links to Excel files
            # Look for .xlsx, .xls, and Excel-related links
            patterns = [r'\.xlsx', r'\.xls', r'excel', r'Excel']
            
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Check if link is an Excel file
                for pattern in patterns:
                    if re.search(pattern, href, re.IGNORECASE) or re.search(pattern, text, re.IGNORECASE):
                        # Resolve relative URLs
                        if href.startswith('/'):
                            full_url = f"{base_url}{href}"
                        elif not href.startswith('http'):
                            full_url = f"{base_url}/{href}"
                        else:
                            full_url = href
                        
                        # Extract filename
                        filename = href.split('/')[-1]
                        
                        # Use link text as alternative filename if available
                        if text and len(text) < 200:
                            filename = text if text else filename
                        
                        excel_links.append({
                            'url': full_url,
                            'filename': filename,
                            'text': text
                        })
                        break
            
            # Also check for direct file references in iframes or embeds
            for iframe in soup.find_all(['iframe', 'embed']):
                src = iframe.get('src', '')
                if re.search(r'\.xlsx|\.xls', src, re.IGNORECASE):
                    if src.startswith('/'):
                        full_url = f"{base_url}{src}"
                    elif not src.startswith('http'):
                        full_url = f"{base_url}/{src}"
                    else:
                        full_url = src
                    
                    filename = src.split('/')[-1]
                    excel_links.append({
                        'url': full_url,
                        'filename': filename,
                        'text': ''
                    })
        
        except Exception as e:
            logger.error(f"Failed to extract Excel links: {e}")
        
        return excel_links
    
    def download_financial_excel_files(self,
                                      from_date: str,
                                      to_date: str,
                                      period_length: int = 12,
                                      audited_only: bool = True,
                                      output_dir: Union[str, Path] = "financial_excel",
                                      max_files: int = None) -> List[str]:
        """
        Download all Excel files from financial statement announcements
        
        Args:
            from_date: Start date for financial statements
            to_date: End date for financial statements
            period_length: Period length in months (default: 12 for annual)
            audited_only: If True, only download from audited statements
            output_dir: Directory to save downloaded files
            max_files: Maximum number of files to download (None for all)
        
        Returns:
            List of successfully downloaded file paths
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Search for financial statements
        logger.info("Searching for financial statements...")
        announcements = self.search_financial_statements(
            from_date=from_date,
            to_date=to_date,
            period_length=period_length,
            audited_only=audited_only
        )
        
        logger.info(f"Found {len(announcements)} financial statement announcements")
        
        downloaded_files = []
        file_count = 0
        
        for i, announcement in enumerate(announcements):
            try:
                # Skip if max files reached
                if max_files and file_count >= max_files:
                    logger.info(f"Reached maximum file limit ({max_files})")
                    break
                
                # Get announcement URL
                letter_url = None
                if 'Url' in announcement and announcement['Url']:
                    letter_url = announcement['Url']
                    if not letter_url.startswith('http'):
                        letter_url = f"https://codal.ir/{letter_url.lstrip('/')}"
                elif 'TracingNo' in announcement and announcement['TracingNo']:
                    letter_url = f"https://codal.ir/Reports/Decision.aspx?LetterSerial={announcement['TracingNo']}"
                
                if not letter_url:
                    logger.warning("No URL found for announcement")
                    continue
                
                # Fetch announcement page
                logger.info(f"[{i+1}/{len(announcements)}] Fetching: {letter_url}")
                response = self.session.get(letter_url, timeout=self.timeout)
                response.raise_for_status()
                
                # Extract Excel file links
                excel_links = self._extract_excel_links(response.text)
                
                if not excel_links:
                    logger.info(f"No Excel files found in announcement")
                    continue
                
                # Download each Excel file
                for link_info in excel_links:
                    # Skip if max files reached
                    if max_files and file_count >= max_files:
                        break
                    
                    # Generate safe filename
                    symbol = announcement.get('Symbol', 'Unknown')
                    date = announcement.get('PublishDateTime', '').replace('/', '-')[:10]
                    filename = f"{symbol}_{date}_{file_count + 1}_{link_info['filename']}"
                    
                    # Remove invalid characters from filename
                    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                    
                    file_path = output_path / filename
                    
                    # Download file
                    if self._download_file(link_info['url'], file_path):
                        downloaded_files.append(str(file_path))
                        file_count += 1
                    
                    # Small delay between downloads
                    time.sleep(0.5)
                
                # Delay between announcements
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to process announcement: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(downloaded_files)} Excel files to {output_path}")
        return downloaded_files