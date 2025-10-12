"""
Async board members scraper for extracting detailed board information from Codal pages
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import networkx as nx
from crawlee.playwright_crawler import PlaywrightCrawler, PlaywrightCrawlingContext

from .utils import (
    clean_dict, normalize_persian_text, persian_to_english_digits,
    datetime_to_num
)
from .constants import AR_TO_FA_LETTER, FA_TO_EN_DIGITS

# Configure logging
logger = logging.getLogger(__name__)


class BoardMemberScraper:
    """
    Async scraper for extracting board member information from Codal URLs
    
    This scraper uses Playwright to navigate and extract detailed board member
    information including names, national IDs, education, and positions.
    """
    
    def __init__(self, selectors: Dict[str, str] = None):
        """
        Initialize the board member scraper
        
        Args:
            selectors: Custom CSS selectors for extracting information
        """
        self.members_data = []
        self.network = nx.Graph()
        self.failed_urls = []
        
        # Default selectors for Codal board member pages
        self.selectors = selectors or {
            'table': '#dgAssemblyBoardMember > tbody tr:not(.GridHeader)',
            'table_all': '#dgAssemblyBoardMember > tbody tr',
            'company': '#lblCompany',
            'ceo_name': '#lblDMBoardMemberList',
            'ceo_national_id': '#lblDMBoardMemberNationalCode',
            'ceo_degree': '#lblDMBoardMemberDegreeLevel',
            'ceo_major': '#lblDMBoardMemberEducationField',
            'date': '#txbSessionDate',
            'prev_version': '#ucNavigateToNextPrevLetter_hlPrevVersion',
            'next_version': '#ucNavigateToNextPrevLetter_hlNewVersion',
            'assembly_date': '#lblAssemblyDate'
        }
        
        self.crawler = None
    
    async def initialize_crawler(self):
        """Initialize the Playwright crawler"""
        from datetime import timedelta
        
        self.crawler = PlaywrightCrawler(
            headless=True,
            browser_type='chromium',
            max_requests_per_crawl=100,
            request_handler_timeout=timedelta(seconds=60),
        )
        
        # Register the handler
        @self.crawler.router.default_handler
        async def handler(context: PlaywrightCrawlingContext) -> None:
            await self.request_handler(context)
    
    async def request_handler(self, context: PlaywrightCrawlingContext) -> None:
        """
        Handle each URL request
        
        Args:
            context: Playwright crawling context
        """
        url = context.request.url
        logger.info(f"Processing {url}")
        
        try:
            # Navigate to the page
            await context.page.goto(url, timeout=30000, wait_until='domcontentloaded')
            
            # Wait a bit for dynamic content to load
            await context.page.wait_for_timeout(2000)
            
            # Check if the table exists (don't wait for visibility)
            table_exists = await context.page.query_selector(self.selectors['table'])
            
            if table_exists:
                # Scrape the board members
                await self.scrape_board_members(context)
            else:
                logger.warning(f"No board member table found at {url}")
                self.failed_urls.append(url)
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            self.failed_urls.append(url)
    
    async def scrape_board_members(self, context: PlaywrightCrawlingContext) -> None:
        """
        Scrape board member information from the page
        
        Args:
            context: Playwright crawling context
        """
        try:
            # Extract all required information
            rows = await context.page.query_selector_all(self.selectors['table'])
            
            # Extract company and CEO information
            company_symbol = await self._extract_text(context.page, self.selectors['company'])
            ceo_name = await self._extract_text(context.page, self.selectors['ceo_name'])
            ceo_national_id = await self._extract_text(context.page, self.selectors['ceo_national_id'])
            ceo_degree = await self._extract_text(context.page, self.selectors['ceo_degree'])
            ceo_major = await self._extract_text(context.page, self.selectors['ceo_major'])
            
            # Extract dates and navigation info
            date = await self._extract_text(context.page, self.selectors['date'])
            assembly_date = await self._extract_text(context.page, self.selectors['assembly_date'])
            has_prev = await self._element_exists(context.page, self.selectors['prev_version'])
            has_next = await self._element_exists(context.page, self.selectors['next_version'])
            
            # Process date
            date_num = ""
            month = ""
            year = ""
            if date:
                date_num = str(datetime_to_num(date))[:8]
                if date_num:
                    year = date_num[:4]
                    month = date_num[4:6]
            
            # Normalize company symbol
            if company_symbol:
                company_symbol = normalize_persian_text(company_symbol)
            else:
                company_symbol = "Unknown"
            
            # Process assembly date
            if assembly_date:
                assembly_date = persian_to_english_digits(assembly_date).replace('/', '')
            
            # Process each board member row
            for row in rows:
                try:
                    member_info = await row.inner_text()
                    
                    # Skip empty rows or rows with insufficient data
                    if not member_info or member_info.strip() == '':
                        continue
                    
                    info = self._extract_member_info(member_info)
                    
                    if not info or len(info) < 12:
                        logger.debug(f"Skipping incomplete row with {len(info) if info else 0} fields")
                        continue
                    
                    # Create member data dictionary
                    member_data = {
                        'year': year,
                        'date': date_num,
                        'month': month,
                        'assembly_date': assembly_date,
                        'company': company_symbol,
                        'has_previous': has_prev,
                        'has_next': has_next,
                        'url': context.request.url,
                        'prev_member': normalize_persian_text(info[0]) if info[0] else "",
                        'new_member': normalize_persian_text(info[1]) if info[1] else "",
                        'member_id': info[2].strip() if info[2] else "",
                        'prev_representative': normalize_persian_text(info[3]) if info[3] else "",
                        'new_representative': normalize_persian_text(info[4]) if info[4] else "",
                        'national_id': info[5].strip() if info[5] else "",
                        'position': normalize_persian_text(info[6]) if info[6] else "",
                        'is_independent': info[7].strip() == 'غیر موظف' if info[7] else False,
                        'degree': normalize_persian_text(info[8]) if info[8] else "",
                        'major': normalize_persian_text(info[9]) if info[9] else "",
                        'has_multiple_executive': info[10].strip() == 'بله' if info[10] else False,
                        'has_multiple_non_executive': info[11].strip() == 'بله' if info[11] else False,
                        'ceo_name': normalize_persian_text(ceo_name) if ceo_name else "",
                        'ceo_national_id': ceo_national_id.strip() if ceo_national_id else "",
                        'ceo_degree': normalize_persian_text(ceo_degree) if ceo_degree else "",
                        'ceo_major': normalize_persian_text(ceo_major) if ceo_major else "",
                        'scrape_timestamp': datetime.now().isoformat()
                    }
                    
                    self.members_data.append(member_data)
                    
                    # Add to network graph if new member exists
                    if member_data['new_member']:
                        self._update_network(member_data)
                    
                except Exception as e:
                    logger.error(f"Failed to extract row data at {context.request.url}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to scrape board members at {context.request.url}: {e}")
    
    async def _extract_text(self, page, selector: str) -> Optional[str]:
        """
        Extract text from an element
        
        Args:
            page: Playwright page object
            selector: CSS selector
            
        Returns:
            Extracted text or None
        """
        try:
            element = await page.query_selector(selector)
            if element:
                text = await element.inner_text()
                return text.strip() if text else None
            return None
        except Exception as e:
            logger.warning(f"Failed to extract text for selector {selector}: {e}")
            return None
    
    async def _element_exists(self, page, selector: str) -> bool:
        """
        Check if an element exists on the page
        
        Args:
            page: Playwright page object
            selector: CSS selector
            
        Returns:
            True if element exists, False otherwise
        """
        try:
            element = await page.query_selector(selector)
            return element is not None
        except Exception:
            return False
    
    def _extract_member_info(self, member_info: str) -> Optional[List[str]]:
        """
        Extract member information from tab-separated string
        
        Args:
            member_info: Tab-separated member information
            
        Returns:
            List of member info fields or None
        """
        try:
            parts = member_info.split('\t')
            return parts if len(parts) >= 12 else None
        except Exception as e:
            logger.error(f"Error parsing member info: {e}")
            return None
    
    def _update_network(self, member_data: Dict):
        """
        Update the network graph with member connections
        
        Args:
            member_data: Board member data dictionary
        """
        try:
            company = member_data['company']
            member = member_data['new_member']
            
            # Add nodes
            if company not in self.network:
                self.network.add_node(company, node_type='company')
            
            if member and member not in self.network:
                self.network.add_node(
                    member,
                    node_type='person',
                    degree=member_data.get('degree', ''),
                    major=member_data.get('major', '')
                )
            
            # Add edge
            if company and member:
                self.network.add_edge(
                    company,
                    member,
                    position=member_data.get('position', ''),
                    year=member_data.get('year', ''),
                    is_independent=member_data.get('is_independent', False)
                )
                
        except Exception as e:
            logger.error(f"Error updating network: {e}")
    
    async def scrape_urls(self, urls: Union[List[str], pd.DataFrame]) -> pd.DataFrame:
        """
        Scrape board member information from a list of URLs
        
        Args:
            urls: List of URLs or DataFrame with 'url' column
            
        Returns:
            DataFrame with scraped board member information
        """
        # Convert DataFrame to list if needed
        if isinstance(urls, pd.DataFrame):
            if 'url' in urls.columns:
                urls = urls['url'].dropna().tolist()
            else:
                raise ValueError("DataFrame must have 'url' column")
        
        # Filter valid URLs
        urls = [url for url in urls if url and url.startswith('http')]
        
        if not urls:
            logger.warning("No valid URLs to scrape")
            return pd.DataFrame()
        
        logger.info(f"Starting to scrape {len(urls)} URLs")
        
        # Initialize crawler if not already done
        if not self.crawler:
            await self.initialize_crawler()
        
        # Run the crawler
        await self.crawler.run(urls)
        
        # Log statistics
        logger.info(f"Scraped {len(self.members_data)} board member records")
        if self.failed_urls:
            logger.warning(f"Failed to scrape {len(self.failed_urls)} URLs")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.members_data)
        return df
    
    async def scrape_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Scrape board member information from URLs in a CSV file
        
        Args:
            csv_path: Path to CSV file containing URLs
            
        Returns:
            DataFrame with scraped board member information
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            return await self.scrape_urls(df)
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return pd.DataFrame()
    
    def export_to_excel(self, 
                       df: pd.DataFrame = None, 
                       file_path: str = 'board_members.xlsx',
                       include_summary: bool = True) -> None:
        """
        Export board member data to Excel
        
        Args:
            df: DataFrame to export (uses internal data if None)
            file_path: Output Excel file path
            include_summary: Whether to include summary statistics
        """
        try:
            # Use internal data if no DataFrame provided
            if df is None:
                df = pd.DataFrame(self.members_data)
            
            if df.empty:
                logger.warning("No data to export")
                return
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Write main data
                df.to_excel(writer, sheet_name='Board Members', index=False)
                
                # Add summary if requested
                if include_summary:
                    summary = self._create_summary(df)
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add network statistics if available
                if self.network and self.network.number_of_nodes() > 0:
                    network_stats = self._create_network_stats()
                    network_stats.to_excel(writer, sheet_name='Network Stats', index=False)
            
            logger.info(f"Data exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
    
    def export_to_csv(self, 
                     df: pd.DataFrame = None,
                     file_path: str = 'board_members.csv') -> None:
        """
        Export board member data to CSV
        
        Args:
            df: DataFrame to export (uses internal data if None)
            file_path: Output CSV file path
        """
        try:
            # Use internal data if no DataFrame provided
            if df is None:
                df = pd.DataFrame(self.members_data)
            
            if df.empty:
                logger.warning("No data to export")
                return
            
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"Data exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def _create_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics for the board member data
        
        Args:
            df: Board member DataFrame
            
        Returns:
            Summary DataFrame
        """
        summary = {
            'Total Records': [len(df)],
            'Unique Companies': [df['company'].nunique() if 'company' in df.columns else 0],
            'Unique Board Members': [df['new_member'].nunique() if 'new_member' in df.columns else 0],
            'Date Range': [f"{df['date'].min()} - {df['date'].max()}" if 'date' in df.columns and not df['date'].empty else ""],
            'Independent Members': [df['is_independent'].sum() if 'is_independent' in df.columns else 0],
            'Members with Multiple Executive Roles': [df['has_multiple_executive'].sum() if 'has_multiple_executive' in df.columns else 0],
            'Failed URLs': [len(self.failed_urls)]
        }
        return pd.DataFrame(summary)
    
    def _create_network_stats(self) -> pd.DataFrame:
        """
        Create network statistics DataFrame
        
        Returns:
            Network statistics DataFrame
        """
        stats = {
            'Total Nodes': [self.network.number_of_nodes()],
            'Total Edges': [self.network.number_of_edges()],
            'Companies': [len([n for n, d in self.network.nodes(data=True) if d.get('node_type') == 'company'])],
            'Board Members': [len([n for n, d in self.network.nodes(data=True) if d.get('node_type') == 'person'])],
            'Average Degree': [sum(dict(self.network.degree()).values()) / self.network.number_of_nodes() if self.network.number_of_nodes() > 0 else 0],
            'Network Density': [nx.density(self.network) if self.network.number_of_nodes() > 0 else 0]
        }
        return pd.DataFrame(stats)
    
    def visualize_network(self, output_file: str = 'board_network.html') -> None:
        """
        Create an interactive network visualization
        
        Args:
            output_file: Output HTML file path
        """
        try:
            if not self.network or self.network.number_of_nodes() == 0:
                logger.warning("No network data to visualize")
                return
            
            from pyvis.network import Network
            
            # Create Pyvis network
            net = Network(height='750px', width='100%', notebook=False)
            
            # Add nodes with attributes
            for node, attrs in self.network.nodes(data=True):
                if attrs.get('node_type') == 'company':
                    net.add_node(node, label=node, color='#ff7f0e', size=30, title=f"Company: {node}")
                else:
                    title = f"Person: {node}\nDegree: {attrs.get('degree', 'N/A')}\nMajor: {attrs.get('major', 'N/A')}"
                    net.add_node(node, label=node, color='#1f77b4', size=20, title=title)
            
            # Add edges
            for u, v, attrs in self.network.edges(data=True):
                title = f"Position: {attrs.get('position', 'N/A')}\nYear: {attrs.get('year', 'N/A')}"
                net.add_edge(u, v, title=title)
            
            # Configure physics
            net.barnes_hut()
            
            # Save visualization
            net.save_graph(output_file)
            logger.info(f"Network visualization saved to {output_file}")
            
        except ImportError:
            logger.error("pyvis is required for network visualization. Install with: pip install pyvis")
        except Exception as e:
            logger.error(f"Error creating network visualization: {e}")