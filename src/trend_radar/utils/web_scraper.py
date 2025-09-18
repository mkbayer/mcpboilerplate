"""
Enhanced Multi-Source Web Scraper for Comprehensive Trend Analysis

This enhanced version incorporates diverse data sources based on industry best practices:
- Research publications and academic papers
- Patent filings and IP data
- News articles and media coverage
- Market research reports
- Investment and funding data
- Social media trends and sentiment
- Government reports and policy documents
- Industry analyst reports
- Conference proceedings and presentations
- Technical documentation and standards
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import re
import logging
from urllib.parse import urljoin, urlparse, parse_qs
import time
import random
from dataclasses import dataclass
from enum import Enum

class SourceType(Enum):
    """Types of trend analysis sources"""
    RESEARCH_PUBLICATION = "research_publication"
    PATENT_DATA = "patent_data"
    NEWS_ARTICLE = "news_article"
    MARKET_RESEARCH = "market_research"
    INVESTMENT_DATA = "investment_data"
    SOCIAL_MEDIA = "social_media"
    GOVERNMENT_REPORT = "government_report"
    ANALYST_REPORT = "analyst_report"
    CONFERENCE_PAPER = "conference_paper"
    TECHNICAL_STANDARD = "technical_standard"
    STARTUP_DATABASE = "startup_database"
    JOB_MARKET = "job_market"

@dataclass
class TrendSource:
    """Configuration for a trend analysis source"""
    name: str
    base_url: str
    source_type: SourceType
    search_endpoint: Optional[str] = None
    api_key_required: bool = False
    rate_limit_delay: float = 1.0
    reliability_score: float = 0.8
    search_params: Dict[str, Any] = None
    headers: Dict[str, str] = None

class EnhancedWebScraper:
    """Enhanced web scraper with multiple diverse sources for trend analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced web scraper"""
        self.config = config or {}
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize comprehensive source configuration
        self.sources = self._initialize_sources()
        
        # Rate limiting and request management
        self.request_delays = {}
        self.failed_requests = {}
        self.success_rates = {}
        
        # Data quality and validation
        self.quality_filters = self._setup_quality_filters()
        
        # Results aggregation
        self.scraped_data = {
            "trends": [],
            "source_metrics": {},
            "quality_score": 0.0
        }
    
    def _initialize_sources(self) -> List[TrendSource]:
        """Initialize comprehensive list of trend analysis sources"""
        sources = [
            # Academic and Research Publications
            TrendSource(
                name="arXiv",
                base_url="https://arxiv.org",
                source_type=SourceType.RESEARCH_PUBLICATION,
                search_endpoint="/search/?query={query}&searchtype=all&abstracts=show&order=-announced_date_first&size=50",
                reliability_score=0.9,
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; TrendAnalyzer/1.0; Research)'
                }
            ),
            TrendSource(
                name="IEEE Xplore",
                base_url="https://ieeexplore.ieee.org",
                source_type=SourceType.RESEARCH_PUBLICATION,
                search_endpoint="/search/searchresult.jsp?newsearch=true&queryText={query}",
                reliability_score=0.95,
                rate_limit_delay=2.0
            ),
            TrendSource(
                name="PubMed",
                base_url="https://pubmed.ncbi.nlm.nih.gov",
                source_type=SourceType.RESEARCH_PUBLICATION,
                search_endpoint="/?term={query}&sort=date&size=200",
                reliability_score=0.9
            ),
            TrendSource(
                name="Google Scholar",
                base_url="https://scholar.google.com",
                source_type=SourceType.RESEARCH_PUBLICATION,
                search_endpoint="/scholar?q={query}&as_ylo=2023&scisbd=1",
                reliability_score=0.85,
                rate_limit_delay=3.0
            ),
            
            # Patent and IP Sources
            TrendSource(
                name="Google Patents",
                base_url="https://patents.google.com",
                source_type=SourceType.PATENT_DATA,
                search_endpoint="/?q={query}&oq={query}&after=priority:20230101",
                reliability_score=0.9,
                rate_limit_delay=2.0
            ),
            TrendSource(
                name="USPTO",
                base_url="https://ppubs.uspto.gov",
                source_type=SourceType.PATENT_DATA,
                search_endpoint="/dirsearch-public/searches/searchAdv",
                reliability_score=0.95,
                rate_limit_delay=2.5
            ),
            TrendSource(
                name="WIPO Global Brand Database",
                base_url="https://www3.wipo.int",
                source_type=SourceType.PATENT_DATA,
                search_endpoint="/branddb/en/",
                reliability_score=0.9
            ),
            
            # News and Media Sources
            TrendSource(
                name="TechCrunch",
                base_url="https://techcrunch.com",
                source_type=SourceType.NEWS_ARTICLE,
                search_endpoint="/search/{query}",
                reliability_score=0.8
            ),
            TrendSource(
                name="MIT Technology Review",
                base_url="https://www.technologyreview.com",
                source_type=SourceType.NEWS_ARTICLE,
                search_endpoint="/search?q={query}",
                reliability_score=0.9
            ),
            TrendSource(
                name="Wired",
                base_url="https://www.wired.com",
                source_type=SourceType.NEWS_ARTICLE,
                search_endpoint="/search/?q={query}&sort=score",
                reliability_score=0.85
            ),
            TrendSource(
                name="VentureBeat",
                base_url="https://venturebeat.com",
                source_type=SourceType.NEWS_ARTICLE,
                search_endpoint="/?s={query}",
                reliability_score=0.8
            ),
            TrendSource(
                name="Ars Technica",
                base_url="https://arstechnica.com",
                source_type=SourceType.NEWS_ARTICLE,
                search_endpoint="/search/?ie=UTF-8&q={query}",
                reliability_score=0.85
            ),
            
            # Market Research and Analysis
            TrendSource(
                name="Gartner",
                base_url="https://www.gartner.com",
                source_type=SourceType.ANALYST_REPORT,
                search_endpoint="/en/search?keywords={query}",
                reliability_score=0.95,
                rate_limit_delay=3.0
            ),
            TrendSource(
                name="McKinsey Insights",
                base_url="https://www.mckinsey.com",
                source_type=SourceType.ANALYST_REPORT,
                search_endpoint="/search?q={query}",
                reliability_score=0.95
            ),
            TrendSource(
                name="Deloitte Insights",
                base_url="https://www2.deloitte.com",
                source_type=SourceType.ANALYST_REPORT,
                search_endpoint="/us/en/insights.html?q={query}",
                reliability_score=0.9
            ),
            TrendSource(
                name="PwC",
                base_url="https://www.pwc.com",
                source_type=SourceType.ANALYST_REPORT,
                search_endpoint="/gx/en/search.html?q={query}",
                reliability_score=0.9
            ),
            TrendSource(
                name="BCG",
                base_url="https://www.bcg.com",
                source_type=SourceType.ANALYST_REPORT,
                search_endpoint="/search?query={query}",
                reliability_score=0.9
            ),
            
            # Investment and Funding Data
            TrendSource(
                name="Crunchbase",
                base_url="https://www.crunchbase.com",
                source_type=SourceType.INVESTMENT_DATA,
                search_endpoint="/discover/searches?q={query}",
                reliability_score=0.85,
                rate_limit_delay=2.0
            ),
            TrendSource(
                name="PitchBook",
                base_url="https://pitchbook.com",
                source_type=SourceType.INVESTMENT_DATA,
                search_endpoint="/search?q={query}",
                reliability_score=0.9,
                rate_limit_delay=2.5
            ),
            TrendSource(
                name="AngelList",
                base_url="https://angel.co",
                source_type=SourceType.STARTUP_DATABASE,
                search_endpoint="/search?q={query}",
                reliability_score=0.8
            ),
            
            # Government and Policy Sources
            TrendSource(
                name="NIST",
                base_url="https://www.nist.gov",
                source_type=SourceType.GOVERNMENT_REPORT,
                search_endpoint="/search/?k={query}",
                reliability_score=0.95
            ),
            TrendSource(
                name="NSF",
                base_url="https://www.nsf.gov",
                source_type=SourceType.GOVERNMENT_REPORT,
                search_endpoint="/search/?q={query}",
                reliability_score=0.9
            ),
            TrendSource(
                name="European Commission",
                base_url="https://ec.europa.eu",
                source_type=SourceType.GOVERNMENT_REPORT,
                search_endpoint="/search/?QueryText={query}&swlang=en",
                reliability_score=0.9
            ),
            
            # Conference and Event Sources
            TrendSource(
                name="ACM Digital Library",
                base_url="https://dl.acm.org",
                source_type=SourceType.CONFERENCE_PAPER,
                search_endpoint="/action/doSearch?AllField={query}",
                reliability_score=0.95
            ),
            TrendSource(
                name="Springer",
                base_url="https://link.springer.com",
                source_type=SourceType.RESEARCH_PUBLICATION,
                search_endpoint="/search?query={query}&date-facet-mode=between&facet-start-year=2023",
                reliability_score=0.9
            ),
            
            # Technology and Standards
            TrendSource(
                name="W3C",
                base_url="https://www.w3.org",
                source_type=SourceType.TECHNICAL_STANDARD,
                search_endpoint="/Search/Mail/Public/search?keywords={query}",
                reliability_score=0.95
            ),
            TrendSource(
                name="IETF",
                base_url="https://www.ietf.org",
                source_type=SourceType.TECHNICAL_STANDARD,
                search_endpoint="/search/?q={query}",
                reliability_score=0.9
            ),
            
            # Job Market and Talent Sources
            TrendSource(
                name="Indeed",
                base_url="https://www.indeed.com",
                source_type=SourceType.JOB_MARKET,
                search_endpoint="/jobs?q={query}&sort=date",
                reliability_score=0.75,
                rate_limit_delay=2.0
            ),
            TrendSource(
                name="LinkedIn Jobs",
                base_url="https://www.linkedin.com",
                source_type=SourceType.JOB_MARKET,
                search_endpoint="/jobs/search/?keywords={query}&sortBy=DD",
                reliability_score=0.8,
                rate_limit_delay=3.0
            ),
            
            # Specialized Tech Sources
            TrendSource(
                name="GitHub Trending",
                base_url="https://github.com",
                source_type=SourceType.TECHNICAL_STANDARD,
                search_endpoint="/search?q={query}&type=repositories&s=updated&o=desc",
                reliability_score=0.8
            ),
            TrendSource(
                name="Stack Overflow",
                base_url="https://stackoverflow.com",
                source_type=SourceType.TECHNICAL_STANDARD,
                search_endpoint="/search?q={query}&sort=newest",
                reliability_score=0.75
            ),
            TrendSource(
                name="Hacker News",
                base_url="https://hn.algolia.com",
                source_type=SourceType.NEWS_ARTICLE,
                search_endpoint="/?query={query}&type=story&dateRange=pastYear&sort=byDate",
                reliability_score=0.8
            ),
            
            # Industry-specific sources
            TrendSource(
                name="Bio.News",
                base_url="https://bio.news",
                source_type=SourceType.NEWS_ARTICLE,
                search_endpoint="/search?q={query}",
                reliability_score=0.8
            ),
            TrendSource(
                name="AI News",
                base_url="https://artificialintelligence-news.com",
                source_type=SourceType.NEWS_ARTICLE,
                search_endpoint="/?s={query}",
                reliability_score=0.8
            ),
            TrendSource(
                name="Fintech News",
                base_url="https://fintechnews.org",
                source_type=SourceType.NEWS_ARTICLE,
                search_endpoint="/?s={query}",
                reliability_score=0.8
            )
        ]
        
        return sources
    
    def _setup_quality_filters(self) -> Dict[str, Any]:
        """Setup quality filters for scraped content"""
        return {
            "min_content_length": 100,
            "max_age_days": 730,  # 2 years
            "required_keywords": [],
            "excluded_keywords": ["spam", "advertisement", "click here"],
            "min_reliability_score": 0.6,
            "language_filters": ["en"],
            "content_quality_indicators": [
                "doi:", "arxiv:", "published", "research", "study", 
                "analysis", "findings", "methodology", "conclusion"
            ]
        }
    
    async def scrape_comprehensive_trends(
        self,
        query: str,
        max_results_per_source: int = 20,
        source_types: Optional[List[SourceType]] = None,
        time_range_days: int = 365
    ) -> Dict[str, Any]:
        """
        Scrape trends from multiple diverse sources
        
        Args:
            query: Search query for trends
            max_results_per_source: Maximum results per source
            source_types: Specific source types to include (None for all)
            time_range_days: How far back to look for trends
            
        Returns:
            Comprehensive trend data from all sources
        """
        
        self.logger.info(f"Starting comprehensive trend scraping for: '{query}'")
        
        # Filter sources by type if specified
        active_sources = self.sources
        if source_types:
            active_sources = [s for s in self.sources if s.source_type in source_types]
        
        # Initialize session
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        ) as session:
            self.session = session
            
            # Create scraping tasks for all sources
            scraping_tasks = []
            for source in active_sources:
                task = self._scrape_source(
                    source, query, max_results_per_source, time_range_days
                )
                scraping_tasks.append(task)
            
            # Execute all scraping tasks with progress tracking
            results = []
            for i, task in enumerate(asyncio.as_completed(scraping_tasks)):
                try:
                    result = await task
                    results.append(result)
                    self.logger.info(f"Completed source {i+1}/{len(scraping_tasks)}: {result['source_name']}")
                except Exception as e:
                    self.logger.error(f"Failed to scrape source: {e}")
                    results.append({"source_name": "unknown", "trends": [], "error": str(e)})
        
        # Aggregate and process results
        aggregated_results = self._aggregate_results(results, query)
        
        self.logger.info(f"Scraping completed. Found {len(aggregated_results['trends'])} total trends from {len(results)} sources")
        
        return aggregated_results
    
    async def _scrape_source(
        self,
        source: TrendSource,
        query: str,
        max_results: int,
        time_range_days: int
    ) -> Dict[str, Any]:
        """Scrape trends from a specific source"""
        
        source_result = {
            "source_name": source.name,
            "source_type": source.source_type.value,
            "trends": [],
            "scraped_count": 0,
            "quality_score": 0.0,
            "scrape_time": datetime.now().isoformat(),
            "errors": []
        }
        
        try:
            # Apply rate limiting
            await self._apply_rate_limit(source)
            
            # Build search URL
            search_url = self._build_search_url(source, query)
            if not search_url:
                source_result["errors"].append("Could not build search URL")
                return source_result
            
            # Perform the search
            trends = await self._perform_search(source, search_url, max_results, time_range_days)
            
            # Apply quality filtering
            filtered_trends = self._apply_quality_filters(trends, source)
            
            source_result["trends"] = filtered_trends
            source_result["scraped_count"] = len(filtered_trends)
            source_result["quality_score"] = self._calculate_source_quality_score(source, filtered_trends)
            
            self.logger.debug(f"Scraped {len(filtered_trends)} trends from {source.name}")
            
        except Exception as e:
            error_msg = f"Error scraping {source.name}: {str(e)}"
            self.logger.error(error_msg)
            source_result["errors"].append(error_msg)
        
        return source_result
    
    async def _apply_rate_limit(self, source: TrendSource) -> None:
        """Apply rate limiting for the source"""
        last_request_time = self.request_delays.get(source.name, 0)
        time_since_last = time.time() - last_request_time
        
        if time_since_last < source.rate_limit_delay:
            sleep_time = source.rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.request_delays[source.name] = time.time()
    
    def _build_search_url(self, source: TrendSource, query: str) -> Optional[str]:
        """Build search URL for the source"""
        if not source.search_endpoint:
            return None
        
        try:
            # URL encode the query
            import urllib.parse
            encoded_query = urllib.parse.quote_plus(query)
            
            # Build full URL
            search_path = source.search_endpoint.format(query=encoded_query)
            full_url = urljoin(source.base_url, search_path)
            
            return full_url
            
        except Exception as e:
            self.logger.error(f"Failed to build URL for {source.name}: {e}")
            return None
    
    async def _perform_search(
        self,
        source: TrendSource,
        search_url: str,
        max_results: int,
        time_range_days: int
    ) -> List[Dict[str, Any]]:
        """Perform the actual search and extract trends"""
        
        trends = []
        
        try:
            # Prepare headers
            headers = source.headers or {
                'User-Agent': 'Mozilla/5.0 (compatible; TrendAnalyzer/1.0)'
            }
            
            # Make request
            async with self.session.get(search_url, headers=headers) as response:
                if response.status != 200:
                    self.logger.warning(f"HTTP {response.status} from {source.name}")
                    return trends
                
                content = await response.text()
                
                # Parse content based on source type
                if source.source_type == SourceType.RESEARCH_PUBLICATION:
                    trends = self._parse_research_content(content, source, max_results)
                elif source.source_type == SourceType.PATENT_DATA:
                    trends = self._parse_patent_content(content, source, max_results)
                elif source.source_type == SourceType.NEWS_ARTICLE:
                    trends = self._parse_news_content(content, source, max_results)
                elif source.source_type == SourceType.ANALYST_REPORT:
                    trends = self._parse_analyst_content(content, source, max_results)
                elif source.source_type == SourceType.INVESTMENT_DATA:
                    trends = self._parse_investment_content(content, source, max_results)
                else:
                    trends = self._parse_generic_content(content, source, max_results)
                
                # Add metadata to each trend
                for trend in trends:
                    trend.update({
                        "source_name": source.name,
                        "source_type": source.source_type.value,
                        "source_reliability": source.reliability_score,
                        "scraped_at": datetime.now().isoformat(),
                        "search_url": search_url
                    })
                
        except Exception as e:
            self.logger.error(f"Search failed for {source.name}: {e}")
        
        return trends[:max_results]
    
    def _parse_research_content(self, content: str, source: TrendSource, max_results: int) -> List[Dict[str, Any]]:
        """Parse research publication content"""
        trends = []
        soup = BeautifulSoup(content, 'html.parser')
        
        try:
            if "arxiv" in source.base_url:
                # Parse arXiv results
                entries = soup.find_all('li', class_='arxiv-result')
                for entry in entries[:max_results]:
                    title_elem = entry.find('p', class_='title')
                    abstract_elem = entry.find('span', class_='abstract-short') or entry.find('span', class_='abstract-full')
                    authors_elem = entry.find('p', class_='authors')
                    
                    if title_elem:
                        trend = {
                            "title": title_elem.get_text(strip=True).replace("Title:", "").strip(),
                            "description": abstract_elem.get_text(strip=True) if abstract_elem else "",
                            "authors": authors_elem.get_text(strip=True) if authors_elem else "",
                            "category": "research",
                            "confidence": 0.8,  # High confidence for peer-reviewed research
                            "urls": [urljoin(source.base_url, link.get('href', '')) for link in entry.find_all('a', href=True)[:2]]
                        }
                        trends.append(trend)
            
            elif "ieee" in source.base_url:
                # Parse IEEE Xplore results
                entries = soup.find_all('div', class_='List-results-items')
                for entry in entries[:max_results]:
                    title_elem = entry.find('h2') or entry.find('h3')
                    abstract_elem = entry.find('div', class_='description')
                    
                    if title_elem:
                        trend = {
                            "title": title_elem.get_text(strip=True),
                            "description": abstract_elem.get_text(strip=True) if abstract_elem else "",
                            "category": "research",
                            "confidence": 0.9,  # Very high confidence for IEEE
                            "urls": [urljoin(source.base_url, link.get('href', '')) for link in entry.find_all('a', href=True)[:2]]
                        }
                        trends.append(trend)
            
            else:
                # Generic research content parsing
                articles = soup.find_all(['article', 'div'], class_=re.compile(r'(result|item|paper|publication)', re.I))
                for article in articles[:max_results]:
                    title_elem = article.find(['h1', 'h2', 'h3', 'h4'], string=re.compile(r'.+'))
                    desc_elem = article.find(['p', 'div'], class_=re.compile(r'(abstract|summary|description)', re.I))
                    
                    if title_elem:
                        trend = {
                            "title": title_elem.get_text(strip=True),
                            "description": desc_elem.get_text(strip=True)[:500] if desc_elem else "",
                            "category": "research",
                            "confidence": 0.75,
                            "urls": [urljoin(source.base_url, link.get('href', '')) for link in article.find_all('a', href=True)[:2]]
                        }
                        trends.append(trend)
        
        except Exception as e:
            self.logger.error(f"Failed to parse research content from {source.name}: {e}")
        
        return trends
    
    def _parse_patent_content(self, content: str, source: TrendSource, max_results: int) -> List[Dict[str, Any]]:
        """Parse patent data content"""
        trends = []
        soup = BeautifulSoup(content, 'html.parser')
        
        try:
            if "patents.google" in source.base_url:
                # Parse Google Patents results
                entries = soup.find_all('div', attrs={'data-result': True})
                for entry in entries[:max_results]:
                    title_elem = entry.find('h3')
                    desc_elem = entry.find('div', class_='abstract')
                    inventor_elem = entry.find('div', string=re.compile(r'Inventor', re.I))
                    
                    if title_elem:
                        trend = {
                            "title": title_elem.get_text(strip=True),
                            "description": desc_elem.get_text(strip=True)[:300] if desc_elem else "",
                            "category": "patent",
                            "inventors": inventor_elem.get_text(strip=True) if inventor_elem else "",
                            "confidence": 0.85,  # Patents are factual but may not indicate trends
                            "urls": [urljoin(source.base_url, link.get('href', '')) for link in entry.find_all('a', href=True)[:1]]
                        }
                        trends.append(trend)
            
            else:
                # Generic patent content parsing
                patents = soup.find_all(['div', 'article'], class_=re.compile(r'(patent|result|item)', re.I))
                for patent in patents[:max_results]:
                    title_elem = patent.find(['h1', 'h2', 'h3', 'h4'])
                    desc_elem = patent.find(['p', 'div'], string=re.compile(r'(abstract|summary)', re.I))
                    
                    if title_elem:
                        trend = {
                            "title": title_elem.get_text(strip=True),
                            "description": desc_elem.get_text(strip=True)[:300] if desc_elem else "",
                            "category": "patent", 
                            "confidence": 0.8,
                            "urls": [urljoin(source.base_url, link.get('href', '')) for link in patent.find_all('a', href=True)[:1]]
                        }
                        trends.append(trend)
        
        except Exception as e:
            self.logger.error(f"Failed to parse patent content from {source.name}: {e}")
        
        return trends
    
    def _parse_news_content(self, content: str, source: TrendSource, max_results: int) -> List[Dict[str, Any]]:
        """Parse news article content"""
        trends = []
        soup = BeautifulSoup(content, 'html.parser')
        
        try:
            # Common news article selectors
            article_selectors = [
                'article',
                '[class*="post"]',
                '[class*="article"]', 
                '[class*="story"]',
                '[class*="item"]',
                '[class*="result"]'
            ]
            
            articles = []
            for selector in article_selectors:
                found = soup.select(selector)
                if found:
                    articles = found
                    break
            
            for article in articles[:max_results]:
                # Try multiple title selectors
                title_elem = None
                for title_sel in ['h1', 'h2', 'h3', '.title', '[class*="title"]', '[class*="headline"]']:
                    title_elem = article.select_one(title_sel)
                    if title_elem:
                        break
                
                # Try multiple description selectors
                desc_elem = None
                for desc_sel in ['.excerpt', '.summary', '.description', 'p']:
                    desc_elem = article.select_one(desc_sel)
                    if desc_elem and len(desc_elem.get_text(strip=True)) > 50:
                        break
                
                # Try to find date
                date_elem = article.select_one('[class*="date"], [class*="time"], time')
                
                if title_elem:
                    trend = {
                        "title": title_elem.get_text(strip=True),
                        "description": desc_elem.get_text(strip=True)[:400] if desc_elem else "",
                        "category": "news",
                        "published_date": date_elem.get_text(strip=True) if date_elem else "",
                        "confidence": 0.7,
                        "urls": [urljoin(source.base_url, link.get('href', '')) for link in article.find_all('a', href=True)[:1]]
                    }
                    trends.append(trend)
        
        except Exception as e:
            self.logger.error(f"Failed to parse news content from {source.name}: {e}")
        
        return trends
    
    def _parse_analyst_content(self, content: str, source: TrendSource, max_results: int) -> List[Dict[str, Any]]:
        """Parse analyst report content"""
        trends = []
        soup = BeautifulSoup(content, 'html.parser')
        
        try:
            # Analyst reports often have specific structures
            report_selectors = [
                '[class*="insight"]',
                '[class*="report"]',
                '[class*="research"]',
                '[class*="publication"]',
                'article',
                '[class*="result"]'
            ]
            
            reports = []
            for selector in report_selectors:
                found = soup.select(selector)
                if found:
                    reports = found
                    break
            
            for report in reports[:max_results]:
                title_elem = report.select_one('h1, h2, h3, .title, [class*="title"]')
                desc_elem = report.select_one('.summary, .excerpt, .description, p')
                author_elem = report.select_one('[class*="author"], [class*="analyst"]')
                date_elem = report.select_one('[class*="date"], time')
                
                if title_elem:
                    trend = {
                        "title": title_elem.get_text(strip=True),
                        "description": desc_elem.get_text(strip=True)[:500] if desc_elem else "",
                        "category": "analyst_report",
                        "author": author_elem.get_text(strip=True) if author_elem else "",
                        "published_date": date_elem.get_text(strip=True) if date_elem else "",
                        "confidence": 0.9,  # High confidence for professional analyst reports
                        "urls": [urljoin(source.base_url, link.get('href', '')) for link in report.find_all('a', href=True)[:1]]
                    }
                    trends.append(trend)
        
        except Exception as e:
            self.logger.error(f"Failed to parse analyst content from {source.name}: {e}")
        
        return trends
    
    def _parse_investment_content(self, content: str, source: TrendSource, max_results: int) -> List[Dict[str, Any]]:
        """Parse investment and funding data content"""
        trends = []
        soup = BeautifulSoup(content, 'html.parser')
        
        try:
            # Investment platforms have specific structures
            if "crunchbase" in source.base_url:
                # Parse Crunchbase results
                entries = soup.select('[class*="search-result"], [class*="entity-card"]')
                for entry in entries[:max_results]:
                    title_elem = entry.select_one('h3, h4, .name, [class*="name"]')
                    desc_elem = entry.select_one('.description, .summary')
                    funding_elem = entry.select_one('[class*="funding"], [class*="raised"]')
                    
                    if title_elem:
                        trend = {
                            "title": title_elem.get_text(strip=True),
                            "description": desc_elem.get_text(strip=True)[:300] if desc_elem else "",
                            "category": "investment",
                            "funding_info": funding_elem.get_text(strip=True) if funding_elem else "",
                            "confidence": 0.8,
                            "urls": [urljoin(source.base_url, link.get('href', '')) for link in entry.find_all('a', href=True)[:1]]
                        }
                        trends.append(trend)
            
            else:
                # Generic investment content parsing
                companies = soup.select('[class*="company"], [class*="startup"], [class*="result"]')
                for company in companies[:max_results]:
                    title_elem = company.select_one('h1, h2, h3, .name, [class*="name"]')
                    desc_elem = company.select_one('.description, .summary, p')
                    
                    if title_elem:
                        trend = {
                            "title": title_elem.get_text(strip=True),
                            "description": desc_elem.get_text(strip=True)[:300] if desc_elem else "",
                            "category": "investment",
                            "confidence": 0.75,
                            "urls": [urljoin(source.base_url, link.get('href', '')) for link in company.find_all('a', href=True)[:1]]
                        }
                        trends.append(trend)
        
        except Exception as e:
            self.logger.error(f"Failed to parse investment content from {source.name}: {e}")
        
        return trends
    
    def _parse_generic_content(self, content: str, source: TrendSource, max_results: int) -> List[Dict[str, Any]]:
        """Parse generic content when source type is unknown"""
        trends = []
        soup = BeautifulSoup(content, 'html.parser')
        
        try:
            # Generic selectors that work for most sites
            items = soup.select('article, .result, .item, .post, [class*="search-result"]')
            
            for item in items[:max_results]:
                title_elem = item.select_one('h1, h2, h3, h4, .title, [class*="title"]')
                desc_elem = item.select_one('p, .description, .summary, .excerpt')
                
                if title_elem and len(title_elem.get_text(strip=True)) > 10:
                    trend = {
                        "title": title_elem.get_text(strip=True),
                        "description": desc_elem.get_text(strip=True)[:400] if desc_elem else "",
                        "category": "general",
                        "confidence": 0.6,  # Lower confidence for generic parsing
                        "urls": [urljoin(source.base_url, link.get('href', '')) for link in item.find_all('a', href=True)[:1]]
                    }
                    trends.append(trend)
        
        except Exception as e:
            self.logger.error(f"Failed to parse generic content from {source.name}: {e}")
        
        return trends
    
    def _apply_quality_filters(self, trends: List[Dict[str, Any]], source: TrendSource) -> List[Dict[str, Any]]:
        """Apply quality filters to scraped trends"""
        filtered_trends = []
        
        for trend in trends:
            # Check minimum content length
            title_length = len(trend.get("title", ""))
            desc_length = len(trend.get("description", ""))
            total_length = title_length + desc_length
            
            if total_length < self.quality_filters["min_content_length"]:
                continue
            
            # Check for excluded keywords
            content_text = (trend.get("title", "") + " " + trend.get("description", "")).lower()
            if any(keyword in content_text for keyword in self.quality_filters["excluded_keywords"]):
                continue
            
            # Check source reliability
            if source.reliability_score < self.quality_filters["min_reliability_score"]:
                continue
            
            # Add quality indicators score
            quality_score = 0.0
            for indicator in self.quality_filters["content_quality_indicators"]:
                if indicator.lower() in content_text:
                    quality_score += 0.1
            
            trend["quality_score"] = min(quality_score, 1.0)
            trend["content_length"] = total_length
            
            filtered_trends.append(trend)
        
        # Sort by quality score and content length
        filtered_trends.sort(key=lambda x: (x["quality_score"], x["content_length"]), reverse=True)
        
        return filtered_trends
    
    def _calculate_source_quality_score(self, source: TrendSource, trends: List[Dict[str, Any]]) -> float:
        """Calculate quality score for a source based on scraped trends"""
        if not trends:
            return 0.0
        
        # Base score from source reliability
        base_score = source.reliability_score
        
        # Adjust based on content quality
        avg_quality = sum(trend.get("quality_score", 0.0) for trend in trends) / len(trends)
        avg_length = sum(trend.get("content_length", 0) for trend in trends) / len(trends)
        
        # Normalize length score (optimal around 200-500 characters)
        length_score = min(avg_length / 300.0, 1.0)
        
        # Combined score
        final_score = (base_score * 0.5) + (avg_quality * 0.3) + (length_score * 0.2)
        
        return min(final_score, 1.0)
    
    def _aggregate_results(self, source_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Aggregate results from all sources"""
        
        all_trends = []
        source_metrics = {}
        total_scraped = 0
        successful_sources = 0
        
        # Collect all trends and metrics
        for result in source_results:
            source_name = result["source_name"]
            trends = result.get("trends", [])
            
            all_trends.extend(trends)
            total_scraped += result.get("scraped_count", 0)
            
            source_metrics[source_name] = {
                "trends_count": len(trends),
                "quality_score": result.get("quality_score", 0.0),
                "source_type": result.get("source_type", "unknown"),
                "errors": result.get("errors", []),
                "scrape_time": result.get("scrape_time", "")
            }
            
            if len(trends) > 0:
                successful_sources += 1
        
        # Remove duplicates based on title similarity
        deduplicated_trends = self._remove_duplicate_trends(all_trends)
        
        # Rank and score trends
        ranked_trends = self._rank_trends(deduplicated_trends)
        
        # Calculate overall quality score
        overall_quality = sum(
            metrics["quality_score"] for metrics in source_metrics.values()
        ) / len(source_metrics) if source_metrics else 0.0
        
        # Generate insights about the scraping process
        scraping_insights = self._generate_scraping_insights(source_metrics, ranked_trends)
        
        return {
            "query": query,
            "trends": ranked_trends,
            "total_trends_found": len(ranked_trends),
            "sources_scraped": len(source_results),
            "successful_sources": successful_sources,
            "overall_quality_score": overall_quality,
            "source_metrics": source_metrics,
            "scraping_insights": scraping_insights,
            "scraping_metadata": {
                "scrape_timestamp": datetime.now().isoformat(),
                "total_original_trends": len(all_trends),
                "deduplication_reduction": len(all_trends) - len(deduplicated_trends),
                "source_types_used": list(set(
                    metrics["source_type"] for metrics in source_metrics.values()
                ))
            }
        }
    
    def _remove_duplicate_trends(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate trends based on title similarity"""
        
        if not trends:
            return trends
        
        deduplicated = []
        seen_titles = set()
        
        for trend in trends:
            title = trend.get("title", "").lower().strip()
            
            # Create a simplified version for comparison
            title_words = set(re.findall(r'\b\w+\b', title))
            
            # Check for similarity with existing titles
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(re.findall(r'\b\w+\b', seen_title))
                
                # Calculate Jaccard similarity
                if title_words and seen_words:
                    similarity = len(title_words.intersection(seen_words)) / len(title_words.union(seen_words))
                    if similarity > 0.7:  # 70% similarity threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(trend)
                seen_titles.add(title)
        
        self.logger.info(f"Deduplication: {len(trends)} -> {len(deduplicated)} trends")
        return deduplicated
    
    def _rank_trends(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank trends by relevance and quality"""
        
        for trend in trends:
            # Calculate ranking score based on multiple factors
            confidence = trend.get("confidence", 0.5)
            quality_score = trend.get("quality_score", 0.5)
            source_reliability = trend.get("source_reliability", 0.5)
            content_length = trend.get("content_length", 0)
            
            # Normalize content length (optimal around 200-400 characters)
            length_score = min(content_length / 300.0, 1.0)
            
            # Source type bonuses
            source_type_bonus = {
                "research_publication": 0.2,
                "analyst_report": 0.15,
                "patent_data": 0.1,
                "government_report": 0.15,
                "news_article": 0.05,
                "investment_data": 0.1
            }
            
            type_bonus = source_type_bonus.get(trend.get("source_type", ""), 0.0)
            
            # Calculate final ranking score
            ranking_score = (
                confidence * 0.3 +
                quality_score * 0.25 +
                source_reliability * 0.2 +
                length_score * 0.15 +
                type_bonus * 0.1
            )
            
            trend["ranking_score"] = ranking_score
        
        # Sort by ranking score
        ranked_trends = sorted(trends, key=lambda x: x.get("ranking_score", 0), reverse=True)
        
        # Add rank position
        for i, trend in enumerate(ranked_trends):
            trend["rank"] = i + 1
        
        return ranked_trends
    
    def _generate_scraping_insights(
        self, 
        source_metrics: Dict[str, Any], 
        trends: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate insights about the scraping process and results"""
        
        if not source_metrics or not trends:
            return {"message": "No data available for insights"}
        
        # Source performance analysis
        best_sources = sorted(
            source_metrics.items(),
            key=lambda x: x[1]["quality_score"],
            reverse=True
        )[:3]
        
        worst_sources = [
            name for name, metrics in source_metrics.items()
            if metrics["trends_count"] == 0 or metrics["quality_score"] < 0.3
        ]
        
        # Source type distribution
        source_type_counts = {}
        for metrics in source_metrics.values():
            source_type = metrics["source_type"]
            source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        
        # Trend category distribution
        category_counts = {}
        for trend in trends:
            category = trend.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Quality analysis
        high_quality_trends = len([t for t in trends if t.get("ranking_score", 0) > 0.7])
        avg_confidence = sum(t.get("confidence", 0) for t in trends) / len(trends)
        
        return {
            "best_performing_sources": [name for name, _ in best_sources],
            "poor_performing_sources": worst_sources,
            "source_type_distribution": source_type_counts,
            "trend_category_distribution": category_counts,
            "quality_metrics": {
                "high_quality_trends": high_quality_trends,
                "average_confidence": avg_confidence,
                "quality_percentage": high_quality_trends / len(trends) * 100 if trends else 0
            },
            "recommendations": self._generate_scraping_recommendations(source_metrics, trends)
        }
    
    def _generate_scraping_recommendations(
        self, 
        source_metrics: Dict[str, Any], 
        trends: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for improving scraping results"""
        
        recommendations = []
        
        # Check source diversity
        successful_sources = len([
            m for m in source_metrics.values() if m["trends_count"] > 0
        ])
        
        if successful_sources < 5:
            recommendations.append("Consider adding more data sources for better coverage")
        
        # Check quality
        avg_quality = sum(
            m["quality_score"] for m in source_metrics.values()
        ) / len(source_metrics)
        
        if avg_quality < 0.5:
            recommendations.append("Improve content quality filters or source selection")
        
        # Check category diversity
        categories = set(t.get("category", "unknown") for t in trends)
        if len(categories) < 3:
            recommendations.append("Broaden search terms to capture more trend categories")
        
        # Check for failed sources
        failed_sources = len([
            m for m in source_metrics.values() if m["trends_count"] == 0
        ])
        
        if failed_sources > len(source_metrics) * 0.3:  # More than 30% failed
            recommendations.append("Review and update source configurations - high failure rate")
        
        return recommendations
    
    async def get_source_health_report(self) -> Dict[str, Any]:
        """Get health report of all configured sources"""
        
        health_report = {
            "total_sources": len(self.sources),
            "source_health": {},
            "recommendations": []
        }
        
        # Test each source with a simple query
        test_query = "artificial intelligence"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            self.session = session
            
            for source in self.sources:
                try:
                    search_url = self._build_search_url(source, test_query)
                    if not search_url:
                        health_report["source_health"][source.name] = {
                            "status": "configuration_error",
                            "error": "Cannot build search URL"
                        }
                        continue
                    
                    async with session.get(search_url, headers=source.headers or {}) as response:
                        health_report["source_health"][source.name] = {
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "status_code": response.status,
                            "response_time": response.headers.get("X-Response-Time", "N/A")
                        }
                
                except Exception as e:
                    health_report["source_health"][source.name] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        # Generate recommendations based on health
        healthy_count = len([
            h for h in health_report["source_health"].values() 
            if h.get("status") == "healthy"
        ])
        
        health_percentage = healthy_count / len(self.sources) * 100
        
        if health_percentage < 70:
            health_report["recommendations"].append("Review failed sources and update configurations")
        
        health_report["health_percentage"] = health_percentage
        
        return health_report
    
    def get_supported_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get information about all supported sources"""
        
        sources_by_type = {}
        
        for source in self.sources:
            source_type = source.source_type.value
            if source_type not in sources_by_type:
                sources_by_type[source_type] = []
            
            sources_by_type[source_type].append({
                "name": source.name,
                "base_url": source.base_url,
                "reliability_score": source.reliability_score,
                "rate_limit_delay": source.rate_limit_delay,
                "api_key_required": source.api_key_required
            })
        
        return sources_by_type

# Example usage and integration class
class TrendScrapingOrchestrator:
    """Orchestrator for managing the enhanced web scraping process"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.scraper = EnhancedWebScraper(config)
        self.logger = logging.getLogger(__name__)
    
    async def comprehensive_trend_search(
        self,
        query: str,
        focus_areas: Optional[List[str]] = None,
        time_range_days: int = 365,
        max_results_per_source: int = 15
    ) -> Dict[str, Any]:
        """
        Perform comprehensive trend search across all sources
        
        Args:
            query: Search query
            focus_areas: Specific areas to focus on (e.g., ['technology', 'healthcare'])
            time_range_days: Time range for search
            max_results_per_source: Maximum results per source
            
        Returns:
            Comprehensive trend analysis results
        """
        
        # Determine source types based on focus areas
        source_types = None
        if focus_areas:
            source_type_mapping = {
                'technology': [SourceType.RESEARCH_PUBLICATION, SourceType.PATENT_DATA, SourceType.TECHNICAL_STANDARD],
                'business': [SourceType.ANALYST_REPORT, SourceType.INVESTMENT_DATA, SourceType.MARKET_RESEARCH],
                'academic': [SourceType.RESEARCH_PUBLICATION, SourceType.CONFERENCE_PAPER],
                'industry': [SourceType.NEWS_ARTICLE, SourceType.ANALYST_REPORT, SourceType.MARKET_RESEARCH],
                'policy': [SourceType.GOVERNMENT_REPORT],
                'investment': [SourceType.INVESTMENT_DATA, SourceType.STARTUP_DATABASE],
                'employment': [SourceType.JOB_MARKET]
            }
            
            source_types = []
            for area in focus_areas:
                source_types.extend(source_type_mapping.get(area.lower(), []))
            source_types = list(set(source_types))  # Remove duplicates
        
        # Perform the comprehensive search
        results = await self.scraper.scrape_comprehensive_trends(
            query=query,
            source_types=source_types,
            time_range_days=time_range_days,
            max_results_per_source=max_results_per_source
        )
        
        # Enhance results with additional analysis
        enhanced_results = self._enhance_results(results, query, focus_areas)
        
        return enhanced_results
    
    def _enhance_results(
        self, 
        results: Dict[str, Any], 
        query: str, 
        focus_areas: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Enhance scraping results with additional analysis"""
        
        trends = results.get("trends", [])
        
        # Add trend clustering and categorization
        trend_clusters = self._cluster_trends(trends)
        
        # Add temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(trends)
        
        # Add source diversity analysis
        source_diversity = self._analyze_source_diversity(results.get("source_metrics", {}))
        
        enhanced_results = {
            **results,
            "trend_clusters": trend_clusters,
            "temporal_analysis": temporal_analysis,
            "source_diversity_analysis": source_diversity,
            "search_metadata": {
                "original_query": query,
                "focus_areas": focus_areas or [],
                "enhancement_timestamp": datetime.now().isoformat()
            }
        }
        
        return enhanced_results
    
    def _cluster_trends(self, trends: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster trends by similarity and topic"""
        
        clusters = {}
        
        for trend in trends:
            # Simple clustering based on category and keywords
            category = trend.get("category", "unknown")
            title = trend.get("title", "").lower()
            
            # Identify cluster key based on common keywords
            cluster_key = category
            
            # Look for common technology terms for subclustering
            tech_keywords = ["ai", "machine learning", "blockchain", "quantum", "iot", "5g", "cloud"]
            for keyword in tech_keywords:
                if keyword in title:
                    cluster_key = f"{category}_{keyword.replace(' ', '_')}"
                    break
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            
            clusters[cluster_key].append(trend)
        
        # Sort clusters by size
        return dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))
    
    def _analyze_temporal_patterns(self, trends: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in the trends"""
        
        # This would be enhanced with proper date parsing
        # For now, return basic analysis
        return {
            "total_trends": len(trends),
            "trends_with_dates": len([t for t in trends if t.get("published_date")]),
            "recent_trend_percentage": 75,  # Placeholder
            "trend_velocity": "increasing"  # Placeholder
        }
    
    def _analyze_source_diversity(self, source_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze diversity of sources used"""
        
        if not source_metrics:
            return {"diversity_score": 0, "coverage": "poor"}
        
        source_types = set()
        active_sources = 0
        
        for metrics in source_metrics.values():
            if metrics.get("trends_count", 0) > 0:
                active_sources += 1
                source_types.add(metrics.get("source_type", "unknown"))
        
        diversity_score = len(source_types) / len(SourceType) * active_sources / len(source_metrics)
        
        coverage_rating = "excellent" if diversity_score > 0.7 else "good" if diversity_score > 0.5 else "fair" if diversity_score > 0.3 else "poor"
        
        return {
            "diversity_score": diversity_score,
            "active_source_types": len(source_types),
            "total_source_types": len(SourceType),
            "active_sources": active_sources,
            "coverage_rating": coverage_rating
        }
    