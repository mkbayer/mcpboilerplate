"""
Data Collector Agent - Responsible for gathering trend data from various sources.
"""

from datetime import datetime
from typing import Dict, Any, List
from ..agents.base_agent import MCPAgent
from ..models.mcp_message import AgentCapability
from ..utils.web_scraper import WebScraper


class DataCollectorAgent(MCPAgent):
    """Agent specialized in collecting and aggregating trend data"""
    
    def __init__(self, llm_base_url: str = "http://localhost:11434", model_name: str = "gpt-oss:20b", **kwargs):
        super().__init__("data_collector", llm_base_url=llm_base_url, model_name=model_name)
        self.data_sources = [
            "tech_blogs",
            "news_apis", 
            "social_media_trends",
            "research_papers",
            "patent_databases",
            "startup_databases"
        ]
        
        # Web scraping categories mapping
        self.web_categories_mapping = {
            "technology": ["technology", "ai_ml"],
            "business": ["business", "startup"],
            "social": ["technology", "business"],
            "environmental": ["technology", "business"]
        }
    
    def _define_capabilities(self) -> List[AgentCapability]:
        """Define data collection capabilities"""
        return [
            AgentCapability(
                name="web_scraping",
                description="Extract trend signals from web sources",
                input_types=["url", "query"],
                output_types=["raw_data", "structured_trends"],
                confidence_level=0.8
            ),
            AgentCapability(
                name="api_integration", 
                description="Gather data from various APIs",
                input_types=["api_config", "query"],
                output_types=["json_data", "trend_signals"],
                confidence_level=0.9
            ),
            AgentCapability(
                name="data_extraction",
                description="Extract structured data from unstructured sources",
                input_types=["text", "documents"],
                output_types=["structured_data", "keywords", "entities"],
                confidence_level=0.75
            ),
            AgentCapability(
                name="trend_signal_detection",
                description="Identify early trend signals in raw data",
                input_types=["raw_data", "historical_data"],
                output_types=["trend_signals", "anomalies"],
                confidence_level=0.7
            ),
            AgentCapability(
                name="real_time_data_collection",
                description="Collect real-time trend data from internet sources",
                input_types=["query", "categories"],
                output_types=["web_trends", "article_data"],
                confidence_level=0.85
            )
        ]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main data collection processing pipeline
        
        Args:
            task: Dictionary containing query, sources, and collection parameters
            
        Returns:
            Dictionary with collected and structured trend data
        """
        query = task.get("query", "emerging technology trends")
        sources = task.get("sources", self.data_sources)
        collection_depth = task.get("depth", "standard")  # light, standard, deep
        
        self.logger.info(f"Starting data collection for: '{query}'")
        self.update_progress(0.1, "Initializing data collection")
        
        try:
            # Step 1: Collect real web data using WebScraper
            web_trends = await self._collect_web_trends(query, collection_depth)
            self.update_progress(0.3, "Web trends collected")
            
            # Step 2: Query LLM for additional trend identification and analysis
            llm_trends = await self._identify_trends_via_llm(query, collection_depth)
            self.update_progress(0.5, "LLM trend analysis completed")
            
            # Step 3: Simulate additional data source queries (legacy sources)
            source_data = await self._query_data_sources(query, sources)
            self.update_progress(0.7, "Additional data sources queried")
            
            # Step 4: Merge and structure all collected data
            all_trends = self._merge_trend_sources(web_trends, llm_trends)
            structured_data = await self._structure_collected_data(all_trends, source_data)
            self.update_progress(0.9, "Data structured and validated")
            
            # Ensure structured_data is not None
            if not structured_data:
                self.logger.warning("Structured data is empty, creating default")
                structured_data = self._create_fallback_trends(query)
            
            # Compile final results with enhanced metadata
            collection_result = {
                "raw_trends": structured_data,
                "collection_metadata": {
                    "query": query,
                    "sources_queried": sources,
                    "collection_depth": collection_depth,
                    "timestamp": datetime.now().isoformat(),
                    "trends_found": len(structured_data),
                    "web_trends_count": len(web_trends),
                    "llm_trends_count": len(llm_trends),
                    "confidence_scores": [t.get("confidence", 0.5) for t in structured_data],
                    "collection_methods": ["web_scraping", "llm_analysis", "simulated_sources"]
                },
                "quality_metrics": {
                    "completeness": self._calculate_completeness(structured_data),
                    "diversity": self._calculate_source_diversity(sources),
                    "freshness": self._calculate_freshness(structured_data),
                    "web_data_ratio": len(web_trends) / max(1, len(structured_data))
                }
            }
            
            self.update_progress(1.0, "Data collection completed")
            self.logger.info(f"Collected {len(structured_data)} trends ({len(web_trends)} from web, {len(llm_trends)} from LLM)")
            
            return collection_result
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            # Return a minimal valid result to prevent pipeline failure
            fallback_result = {
                "raw_trends": self._create_fallback_trends(query),
                "collection_metadata": {
                    "query": query,
                    "sources_queried": [],
                    "collection_depth": collection_depth,
                    "timestamp": datetime.now().isoformat(),
                    "trends_found": 1,
                    "error": str(e),
                    "collection_methods": ["fallback"]
                },
                "quality_metrics": {
                    "completeness": 0.3,
                    "diversity": 0.0,
                    "freshness": 1.0,
                    "web_data_ratio": 0.0
                }
            }
            self.logger.info("Returning fallback result to continue pipeline")
            return fallback_result
    
    async def _collect_web_trends(self, query: str, depth: str) -> List[Dict[str, Any]]:
        """Collect real trends from web sources using WebScraper"""
        web_trends = []
        
        try:
            # Determine categories to search based on query
            search_categories = self._determine_search_categories(query)
            
            # Set max articles based on collection depth
            depth_limits = {
                "light": 20,
                "standard": 40, 
                "deep": 80
            }
            max_articles = depth_limits.get(depth, 40)
            
            self.logger.info(f"Collecting web trends: {search_categories}, max_articles: {max_articles}")
            
            # Use WebScraper to collect real data
            async with WebScraper() as scraper:
                web_trends = await scraper.collect_trends(
                    query=query,
                    categories=search_categories,
                    max_articles=max_articles
                )
            
            self.logger.info(f"Web scraper collected {len(web_trends)} trends")
            
            # Enhance web trends with additional metadata
            for trend in web_trends:
                trend['collection_method'] = 'web_scraping'
                trend['data_source'] = 'real_web'
                
                # Add impact assessment based on article metrics
                trend['impact_level'] = self._assess_web_trend_impact(trend)
                trend['time_horizon'] = self._assess_web_trend_timeline(trend)
        
        except Exception as e:
            self.logger.error(f"Web trend collection failed: {e}")
            self.logger.info("Continuing with other collection methods")
        
        return web_trends
    
    def _determine_search_categories(self, query: str) -> List[str]:
        """Determine which web categories to search based on query content"""
        query_lower = query.lower()
        
        categories = []
        
        # Technology indicators
        if any(term in query_lower for term in ['ai', 'artificial intelligence', 'machine learning', 'technology', 'tech', 'digital', 'software', 'hardware', 'blockchain', 'crypto', 'quantum']):
            categories.extend(['technology', 'ai_ml'])
        
        # Business indicators
        if any(term in query_lower for term in ['business', 'market', 'startup', 'company', 'enterprise', 'industry', 'economy', 'finance']):
            categories.extend(['business', 'startup'])
        
        # Default categories if none detected
        if not categories:
            categories = ['technology', 'business']
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(categories))
    
    def _assess_web_trend_impact(self, trend: Dict[str, Any]) -> str:
        """Assess impact level of web trend based on metrics"""
        article_count = trend.get('article_count', 1)
        relevance_score = trend.get('relevance_score', 0)
        
        # Calculate impact based on multiple factors
        if article_count >= 5 and relevance_score >= 10:
            return 'high'
        elif article_count >= 3 and relevance_score >= 5:
            return 'medium'
        else:
            return 'low'
    
    def _assess_web_trend_timeline(self, trend: Dict[str, Any]) -> str:
        """Assess timeline for web trend based on content"""
        title = trend.get('title', '').lower()
        description = trend.get('description', '').lower()
        
        # Look for timeline indicators in text
        if any(term in title + description for term in ['emerging', 'new', 'breakthrough', 'revolutionary']):
            return 'emerging'
        elif any(term in title + description for term in ['future', 'coming', 'next year', 'upcoming']):
            return 'short_term'
        elif any(term in title + description for term in ['trend', 'growing', 'increasing']):
            return 'medium_term'
        else:
            return 'medium_term'  # default
    
    def _merge_trend_sources(self, web_trends: List[Dict[str, Any]], llm_trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge trends from different sources and remove duplicates"""
        merged_trends = []
        
        # Add web trends first (higher priority)
        for trend in web_trends:
            trend['primary_source'] = 'web_scraping'
            merged_trends.append(trend)
        
        # Add LLM trends, checking for duplicates
        for llm_trend in llm_trends:
            llm_trend['primary_source'] = 'llm_analysis'
            
            # Simple duplicate detection based on title similarity
            is_duplicate = False
            llm_title_words = set(llm_trend.get('title', '').lower().split())
            
            for existing_trend in merged_trends:
                existing_title_words = set(existing_trend.get('title', '').lower().split())
                overlap = len(llm_title_words.intersection(existing_title_words))
                
                # If significant word overlap, consider it a duplicate
                if overlap >= 2 and len(llm_title_words) > 2:
                    is_duplicate = True
                    # Enhance existing trend with LLM insights
                    existing_trend['llm_analysis'] = llm_trend.get('description', '')
                    existing_trend['llm_keywords'] = llm_trend.get('keywords', [])
                    break
            
            if not is_duplicate:
                merged_trends.append(llm_trend)
        
        self.logger.info(f"Merged {len(web_trends)} web trends and {len(llm_trends)} LLM trends into {len(merged_trends)} unique trends")
        return merged_trends
    
    def _create_fallback_trends(self, query: str) -> List[Dict[str, Any]]:
        """Create fallback trends when data collection fails"""
        return [{
            'id': f'fallback_trend_{datetime.now().strftime("%Y%m%d")}',
            'title': f'Sample Trend for {query}',
            'description': f'Sample trend data for {query} analysis - generated as fallback',
            'category': 'technology',
            'keywords': [query.split()[0] if query.split() else 'trend', 'innovation', 'technology'],
            'confidence': 0.3,
            'sources': ['fallback'],
            'collected_at': datetime.now().isoformat(),
            'collection_method': 'fallback',
            'impact_level': 'medium',
            'time_horizon': 'medium_term',
            'validation': {
                'title_length_ok': True,
                'has_description': True,
                'has_keywords': True,
                'category_valid': True
            }
        }]
    
    async def _identify_trends_via_llm(self, query: str, depth: str) -> List[Dict[str, Any]]:
        """Use LLM to identify and describe trends"""
        depth_instructions = {
            "light": "Identify 3-4 key trends with brief descriptions",
            "standard": "Identify 5-7 trends with detailed descriptions and context",
            "deep": "Identify 8-12 trends with comprehensive analysis and sub-trends"
        }
        
        system_prompt = """You are a trend analysis expert. Your task is to identify emerging 
        and significant trends based on the given query. Focus on providing actionable, 
        well-researched insights that would be valuable for strategic planning."""
        
        prompt = f"""
        Query: {query}
        
        {depth_instructions.get(depth, depth_instructions['standard'])}
        
        For each trend, provide:
        1. Trend Title (concise and descriptive)
        2. Description (2-3 sentences explaining the trend)
        3. Category (technology/business/social/environmental)
        4. Key Indicators (signals that suggest this trend is emerging)
        5. Potential Impact (what changes this trend might drive)
        6. Time Horizon (emerging/short_term/medium_term/long_term)
        7. Related Keywords (5-7 relevant terms)
        
        Format each trend clearly with headers and maintain consistency.
        """
        
        response = await self.query_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=800 if depth == "deep" else 600
        )
        
        return self._parse_llm_trends_response(response)
    
    def _parse_llm_trends_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured trend data"""
        trends = []
        current_trend = {}
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for trend boundaries (numbers, bullets, or "Trend" keyword)
            if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                line.startswith(('•', '-', '*')) or
                line.lower().startswith('trend')):
                
                # Save previous trend if it has content
                if current_trend and 'title' in current_trend:
                    current_trend['collection_method'] = 'llm_analysis'
                    trends.append(current_trend)
                
                # Start new trend
                title = line.split('.', 1)[-1].strip() if '.' in line else line
                title = title.replace('Trend:', '').replace('trend:', '').strip()
                current_trend = {
                    'title': title,
                    'description': '',
                    'category': 'technology',  # default
                    'indicators': [],
                    'keywords': [],
                    'confidence': 0.6  # default confidence
                }
            
            elif current_trend:
                # Parse content lines
                lower_line = line.lower()
                
                if 'description:' in lower_line or 'desc:' in lower_line:
                    current_trend['description'] = line.split(':', 1)[1].strip()
                elif 'category:' in lower_line or 'cat:' in lower_line:
                    category = line.split(':', 1)[1].strip().lower()
                    if category in ['technology', 'business', 'social', 'environmental']:
                        current_trend['category'] = category
                elif 'keywords:' in lower_line or 'tags:' in lower_line:
                    keywords_str = line.split(':', 1)[1].strip()
                    current_trend['keywords'] = [
                        k.strip() for k in keywords_str.split(',') if k.strip()
                    ]
                elif 'impact:' in lower_line:
                    impact_text = line.split(':', 1)[1].strip().lower()
                    if any(level in impact_text for level in ['high', 'critical']):
                        current_trend['impact_level'] = 'high'
                    elif 'medium' in impact_text:
                        current_trend['impact_level'] = 'medium'
                    else:
                        current_trend['impact_level'] = 'low'
                elif 'time' in lower_line and 'horizon' in lower_line:
                    horizon_text = line.split(':', 1)[1].strip().lower()
                    if 'emerging' in horizon_text:
                        current_trend['time_horizon'] = 'emerging'
                    elif 'short' in horizon_text:
                        current_trend['time_horizon'] = 'short_term'
                    elif 'long' in horizon_text:
                        current_trend['time_horizon'] = 'long_term'
                    else:
                        current_trend['time_horizon'] = 'medium_term'
                elif not any(key in lower_line for key in [':', 'title', 'trend']):
                    # Add to description if no specific field identified
                    if current_trend['description']:
                        current_trend['description'] += ' ' + line
                    else:
                        current_trend['description'] = line
        
        # Don't forget the last trend
        if current_trend and 'title' in current_trend:
            current_trend['collection_method'] = 'llm_analysis'
            trends.append(current_trend)
        
        # Validate and clean up trends
        validated_trends = []
        for trend in trends:
            if trend.get('title') and len(trend.get('title', '')) > 3:
                # Set defaults for missing fields
                trend.setdefault('description', 'No description available')
                trend.setdefault('category', 'technology')
                trend.setdefault('keywords', [])
                trend.setdefault('confidence', 0.6)
                trend.setdefault('impact_level', 'medium')
                trend.setdefault('time_horizon', 'medium_term')
                
                validated_trends.append(trend)
        
        self.logger.debug(f"Parsed {len(validated_trends)} trends from LLM response")
        return validated_trends
    
    async def _query_data_sources(self, query: str, sources: List[str]) -> Dict[str, Any]:
        """Simulate querying various data sources (placeholder for additional real implementations)"""
        # In a real implementation, this would:
        # 1. Query news APIs (NewsAPI, Bing News, etc.)
        # 2. Access social media APIs (Twitter, Reddit, etc.)
        # 3. Query research databases (arXiv, Google Scholar, etc.)
        # 4. Check patent databases (USPTO, EPO, etc.)
        # 5. Access startup databases (Crunchbase, AngelList, etc.)
        
        source_results = {}
        
        for source in sources:
            # Simulate source-specific data
            if source == "tech_blogs":
                source_results[source] = {
                    "articles_found": 15,
                    "sentiment": "positive", 
                    "key_topics": ["AI", "blockchain", "quantum", "sustainability"],
                    "trending_keywords": ["artificial intelligence", "machine learning", "green tech"]
                }
            elif source == "social_media_trends":
                source_results[source] = {
                    "trending_hashtags": ["#AIRevolution", "#GreenTech", "#Web3"],
                    "mention_volume": 1250,
                    "engagement_trend": "increasing"
                }
            elif source == "research_papers":
                source_results[source] = {
                    "papers_found": 8,
                    "recent_publications": True,
                    "citation_trends": "growing",
                    "research_areas": ["machine learning", "renewable energy", "biotechnology"]
                }
        
        self.logger.debug(f"Simulated data collection from {len(sources)} sources")
        return source_results
    
    async def _structure_collected_data(
        self, 
        all_trends: List[Dict[str, Any]], 
        source_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Structure and enrich collected data with source information"""
        
        # Ensure all_trends is not None or empty
        if not all_trends:
            self.logger.warning("No trends provided, creating default trend")
            all_trends = self._create_fallback_trends("default query")
        
        structured_trends = []
        
        for i, trend in enumerate(all_trends):
            # Ensure trend is a dictionary
            if not isinstance(trend, dict):
                self.logger.warning(f"Trend {i} is not a dictionary, skipping")
                continue
            
            # Add unique identifier if not present
            if 'id' not in trend:
                trend['id'] = f"trend_{i+1}_{datetime.now().strftime('%Y%m%d')}"
            
            # Add source enrichment
            existing_sources = trend.get('sources', [])
            additional_sources = list(source_data.keys()) if source_data else []
            trend['sources'] = list(set(existing_sources + additional_sources + ['data_collector']))
            
            # Add timestamps
            if 'collected_at' not in trend:
                trend['collected_at'] = datetime.now().isoformat()
            
            # Calculate enhanced confidence based on source corroboration and collection method
            base_confidence = trend.get('confidence', 0.6)
            
            # Boost confidence for web-scraped data (more reliable)
            if trend.get('collection_method') == 'web_scraping':
                base_confidence = min(0.9, base_confidence + 0.2)
            
            # Boost confidence based on multiple sources
            source_boost = min(0.15, len(trend.get('sources', [])) * 0.03)
            trend['confidence'] = min(0.95, base_confidence + source_boost)
            
            # Add validation flags
            trend['validation'] = {
                'title_length_ok': len(str(trend.get('title', ''))) >= 5,
                'has_description': bool(str(trend.get('description', '')).strip()),
                'has_keywords': len(trend.get('keywords', [])) > 0,
                'category_valid': trend.get('category') in ['technology', 'business', 'social', 'environmental'],
                'has_web_data': trend.get('collection_method') == 'web_scraping',
                'source_diversity': len(set(trend.get('sources', []))) >= 2
            }
            
            structured_trends.append(trend)
        
        # Ensure we always return at least one trend
        if not structured_trends:
            structured_trends = self._create_fallback_trends("fallback query")
        
        self.logger.debug(f"Structured {len(structured_trends)} trends")
        return structured_trends
    
    def _calculate_completeness(self, trends: List[Dict[str, Any]]) -> float:
        """Calculate data completeness score"""
        if not trends:
            return 0.0
        
        total_fields = 0
        filled_fields = 0
        
        required_fields = ['title', 'description', 'category', 'keywords']
        
        for trend in trends:
            for field in required_fields:
                total_fields += 1
                if trend.get(field) and str(trend[field]).strip():
                    filled_fields += 1
        
        return round(filled_fields / total_fields, 2) if total_fields > 0 else 0.0
    
    def _calculate_source_diversity(self, sources: List[str]) -> float:
        """Calculate diversity of data sources"""
        max_sources = len(self.data_sources)
        return round(len(sources) / max_sources, 2)
    
    def _calculate_freshness(self, trends: List[Dict[str, Any]]) -> float:
        """Calculate data freshness score (all data is fresh since just collected)"""
        return 1.0  # Perfect freshness for newly collected data
    