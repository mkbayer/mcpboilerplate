"""
Real web scraping utilities for collecting trend data from internet sources.
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
import feedparser
from .logger import get_logger

logger = get_logger(__name__)


class WebScraper:
    """Web scraper for collecting trend data from various sources"""
    
    def __init__(self, timeout: int = 30, max_concurrent: int = 5):
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.session = None
        
        # Define reliable sources for different categories
        self.sources = {
            "technology": {
                "hacker_news": "https://news.ycombinator.com/",
                "tech_crunch_rss": "https://techcrunch.com/feed/",
                "ars_technica_rss": "https://feeds.arstechnica.com/arstechnica/index",
                "wired_rss": "https://www.wired.com/feed/rss",
                "verge_rss": "https://www.theverge.com/rss/index.xml"
            },
            "business": {
                "reuters_tech": "https://www.reuters.com/technology/",
                "bloomberg_rss": "https://feeds.bloomberg.com/technology/news.rss",
                "fortune_rss": "https://fortune.com/section/fortune500/feed/",
                "business_insider_rss": "https://feeds.businessinsider.com/custom/all"
            },
            "ai_ml": {
                "towards_data_science": "https://towardsdatascience.com/feed",
                "ai_news": "https://artificialintelligence-news.com/feed/",
                "machine_learning_mastery": "https://machinelearningmastery.com/feed/",
                "mit_ai": "https://news.mit.edu/rss/topic/artificial-intelligence2"
            },
            "startup": {
                "product_hunt": "https://www.producthunt.com/",
                "crunchbase_news": "https://news.crunchbase.com/feed/",
                "startup_news": "https://news.ycombinator.com/"
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.aclose()
    
    async def collect_trends(self, query: str, categories: List[str] = None, max_articles: int = 50) -> List[Dict[str, Any]]:
        """
        Collect trend data from real internet sources
        
        Args:
            query: Search query for trend analysis
            categories: List of categories to search (technology, business, ai_ml, startup)
            max_articles: Maximum number of articles to collect
            
        Returns:
            List of trend data collected from web sources
        """
        if not categories:
            categories = ["technology", "business", "ai_ml"]
        
        logger.info(f"Collecting trends for query: '{query}' from {len(categories)} categories")
        
        all_articles = []
        
        # Collect from RSS feeds
        rss_articles = await self._collect_from_rss_feeds(categories, max_articles // 2)
        all_articles.extend(rss_articles)
        
        # Collect from web scraping
        scraped_articles = await self._collect_from_web_scraping(query, categories, max_articles // 2)
        all_articles.extend(scraped_articles)
        
        # Filter and rank articles by relevance to query
        relevant_articles = self._filter_by_relevance(all_articles, query)
        
        # Convert to trend format
        trends = self._articles_to_trends(relevant_articles, query)
        
        logger.info(f"Collected {len(trends)} relevant trends from {len(all_articles)} articles")
        return trends
    
    async def _collect_from_rss_feeds(self, categories: List[str], max_articles: int) -> List[Dict[str, Any]]:
        """Collect articles from RSS feeds"""
        articles = []
        
        for category in categories:
            if category not in self.sources:
                continue
            
            category_sources = self.sources[category]
            rss_sources = {k: v for k, v in category_sources.items() if 'rss' in k or 'feed' in v}
            
            for source_name, rss_url in rss_sources.items():
                try:
                    logger.debug(f"Fetching RSS feed: {source_name}")
                    
                    # Use feedparser to parse RSS feeds
                    feed = await asyncio.get_event_loop().run_in_executor(
                        None, feedparser.parse, rss_url
                    )
                    
                    if feed.entries:
                        for entry in feed.entries[:max_articles // len(rss_sources)]:
                            article = {
                                'title': entry.get('title', ''),
                                'description': entry.get('summary', ''),
                                'url': entry.get('link', ''),
                                'published': entry.get('published', ''),
                                'source': source_name,
                                'category': category,
                                'content_type': 'rss'
                            }
                            articles.append(article)
                        
                        logger.debug(f"Collected {len(feed.entries)} articles from {source_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch RSS feed {source_name}: {e}")
                    continue
        
        return articles
    
    async def _collect_from_web_scraping(self, query: str, categories: List[str], max_articles: int) -> List[Dict[str, Any]]:
        """Collect articles via web scraping"""
        articles = []
        
        # Scrape Hacker News
        if "technology" in categories:
            hn_articles = await self._scrape_hacker_news(query, max_articles // 3)
            articles.extend(hn_articles)
        
        # Search specific sites
        search_articles = await self._search_web_sources(query, categories, max_articles // 3)
        articles.extend(search_articles)
        
        return articles
    
    async def _scrape_hacker_news(self, query: str, max_articles: int) -> List[Dict[str, Any]]:
        """Scrape Hacker News for relevant articles"""
        articles = []
        
        try:
            # Search Hacker News
            search_url = f"https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage={max_articles}"
            
            response = await self.session.get(search_url)
            response.raise_for_status()
            data = response.json()
            
            for hit in data.get('hits', []):
                if hit.get('title') and hit.get('url'):
                    article = {
                        'title': hit['title'],
                        'description': hit.get('story_text', '')[:200] + '...' if hit.get('story_text') else '',
                        'url': hit['url'],
                        'published': hit.get('created_at', ''),
                        'source': 'hacker_news',
                        'category': 'technology',
                        'content_type': 'scraped',
                        'score': hit.get('points', 0),
                        'comments': hit.get('num_comments', 0)
                    }
                    articles.append(article)
            
            logger.debug(f"Collected {len(articles)} articles from Hacker News")
            
        except Exception as e:
            logger.warning(f"Failed to scrape Hacker News: {e}")
        
        return articles
    
    async def _search_web_sources(self, query: str, categories: List[str], max_articles: int) -> List[Dict[str, Any]]:
        """Search web sources for relevant content"""
        articles = []
        
        # Use DuckDuckGo search (no API key required)
        try:
            # Search for recent articles
            search_queries = [
                f"{query} trends 2024",
                f"{query} emerging technology",
                f"{query} market analysis",
                f"{query} future predictions"
            ]
            
            for search_query in search_queries[:2]:  # Limit searches
                search_results = await self._duckduckgo_search(search_query, max_articles // 4)
                articles.extend(search_results)
            
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
        
        return articles
    
    async def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform DuckDuckGo search and extract results"""
        articles = []
        
        try:
            # DuckDuckGo Instant Answer API (limited but free)
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            
            response = await self.session.get(search_url)
            response.raise_for_status()
            data = response.json()
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:max_results]:
                if isinstance(topic, dict) and topic.get('Text') and topic.get('FirstURL'):
                    article = {
                        'title': topic['Text'].split(' - ')[0] if ' - ' in topic['Text'] else topic['Text'][:100],
                        'description': topic['Text'],
                        'url': topic['FirstURL'],
                        'published': '',
                        'source': 'duckduckgo_search',
                        'category': 'general',
                        'content_type': 'search_result'
                    }
                    articles.append(article)
            
            logger.debug(f"Found {len(articles)} results from DuckDuckGo")
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        return articles
    
    def _filter_by_relevance(self, articles: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter articles by relevance to the query"""
        query_words = set(query.lower().split())
        relevant_articles = []
        
        for article in articles:
            # Calculate relevance score
            title_words = set(article.get('title', '').lower().split())
            desc_words = set(article.get('description', '').lower().split())
            
            title_overlap = len(query_words.intersection(title_words))
            desc_overlap = len(query_words.intersection(desc_words))
            
            # Score based on word overlap and other factors
            relevance_score = (title_overlap * 3 + desc_overlap * 1)
            
            # Boost score for recent articles
            if self._is_recent(article.get('published', '')):
                relevance_score += 2
            
            # Boost score for high-quality sources
            if article.get('source') in ['tech_crunch_rss', 'wired_rss', 'hacker_news']:
                relevance_score += 1
            
            if relevance_score > 0:
                article['relevance_score'] = relevance_score
                relevant_articles.append(article)
        
        # Sort by relevance score
        relevant_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_articles[:30]  # Return top 30 most relevant
    
    def _is_recent(self, published: str) -> bool:
        """Check if article was published recently (within last 30 days)"""
        if not published:
            return False
        
        try:
            # Try to parse various date formats
            from dateutil import parser
            pub_date = parser.parse(published)
            return (datetime.now() - pub_date.replace(tzinfo=None)) < timedelta(days=30)
        except:
            return False
    
    def _articles_to_trends(self, articles: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Convert collected articles to trend format"""
        trends = []
        
        # Group articles by similar topics
        topic_groups = self._group_articles_by_topic(articles)
        
        for i, (topic, topic_articles) in enumerate(topic_groups.items()):
            if len(topic_articles) < 2:  # Skip topics with too few articles
                continue
            
            trend = {
                'id': f'web_trend_{i+1}',
                'title': self._generate_trend_title(topic, topic_articles),
                'description': self._generate_trend_description(topic_articles),
                'category': self._determine_category(topic_articles),
                'keywords': self._extract_keywords(topic_articles, query),
                'confidence': min(0.9, 0.5 + len(topic_articles) * 0.1),
                'sources': [article['source'] for article in topic_articles],
                'urls': [article['url'] for article in topic_articles[:3]],  # Top 3 URLs
                'collected_at': datetime.now().isoformat(),
                'relevance_score': sum(article.get('relevance_score', 0) for article in topic_articles),
                'article_count': len(topic_articles)
            }
            trends.append(trend)
        
        return trends
    
    def _group_articles_by_topic(self, articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group articles by similar topics using keyword matching"""
        topic_groups = {}
        
        for article in articles:
            title = article.get('title', '').lower()
            
            # Extract key terms from title
            key_terms = self._extract_key_terms(title)
            
            # Find best matching existing topic or create new one
            best_topic = None
            max_overlap = 0
            
            for existing_topic in topic_groups.keys():
                overlap = len(set(key_terms).intersection(set(existing_topic.split())))
                if overlap > max_overlap and overlap >= 2:
                    max_overlap = overlap
                    best_topic = existing_topic
            
            if best_topic:
                topic_groups[best_topic].append(article)
            else:
                # Create new topic group
                topic_name = ' '.join(key_terms[:3]) if key_terms else title[:50]
                topic_groups[topic_name] = [article]
        
        return topic_groups
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        # Extract words (alphanumeric, longer than 2 chars)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        return key_terms[:10]  # Return top 10 key terms
    
    def _generate_trend_title(self, topic: str, articles: List[Dict[str, Any]]) -> str:
        """Generate a trend title from topic and articles"""
        # Use the most common meaningful words
        all_titles = ' '.join([article.get('title', '') for article in articles])
        key_terms = self._extract_key_terms(all_titles)
        
        # Create a descriptive title
        if len(key_terms) >= 2:
            return f"{key_terms[0].title()} {key_terms[1].title()} Developments"
        elif key_terms:
            return f"{key_terms[0].title()} Innovation Trends"
        else:
            return topic.title()
    
    def _generate_trend_description(self, articles: List[Dict[str, Any]]) -> str:
        """Generate trend description from articles"""
        descriptions = [article.get('description', '') for article in articles if article.get('description')]
        
        if descriptions:
            # Take the first substantial description
            for desc in descriptions:
                if len(desc) > 50:
                    return desc[:200] + '...' if len(desc) > 200 else desc
        
        return f"Emerging trend identified from {len(articles)} recent articles and discussions."
    
    def _determine_category(self, articles: List[Dict[str, Any]]) -> str:
        """Determine trend category from articles"""
        categories = [article.get('category', 'technology') for article in articles]
        
        # Return most common category
        from collections import Counter
        category_counts = Counter(categories)
        return category_counts.most_common(1)[0][0] if category_counts else 'technology'
    
    def _extract_keywords(self, articles: List[Dict[str, Any]], query: str) -> List[str]:
        """Extract keywords from articles"""
        all_text = ' '.join([
            article.get('title', '') + ' ' + article.get('description', '')
            for article in articles
        ])
        
        key_terms = self._extract_key_terms(all_text)
        query_terms = query.lower().split()
        
        # Combine and deduplicate
        keywords = list(set(key_terms[:5] + query_terms))
        
        return keywords[:8]  # Return top 8 keywords
    