"""
Data Collector Agent - Responsible for gathering trend data from various sources.
"""

from datetime import datetime
from typing import Dict, Any, List
from ..agents.base_agent import MCPAgent
from ..models.mcp_message import AgentCapability


class DataCollectorAgent(MCPAgent):
    """Agent specialized in collecting and aggregating trend data"""
    
    def __init__(self, **kwargs):
        super().__init__("data_collector", **kwargs)
        self.data_sources = [
            "tech_blogs",
            "news_apis", 
            "social_media_trends",
            "research_papers",
            "patent_databases",
            "startup_databases"
        ]
    
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
        
        # Step 1: Query LLM for trend identification
        raw_trends = await self._identify_trends_via_llm(query, collection_depth)
        self.update_progress(0.4, "Trends identified via LLM")
        
        # Step 2: Simulate data source queries (in production, would hit real APIs)
        source_data = await self._query_data_sources(query, sources)
        self.update_progress(0.7, "Data sources queried")
        
        # Step 3: Structure and validate collected data
        structured_data = await self._structure_collected_data(raw_trends, source_data)
        self.update_progress(0.9, "Data structured and validated")
        
        # Compile final results
        collection_result = {
            "raw_trends": structured_data,
            "collection_metadata": {
                "query": query,
                "sources_queried": sources,
                "collection_depth": collection_depth,
                "timestamp": datetime.now().isoformat(),
                "trends_found": len(structured_data),
                "confidence_scores": [t.get("confidence", 0.5) for t in structured_data]
            },
            "quality_metrics": {
                "completeness": self._calculate_completeness(structured_data),
                "diversity": self._calculate_source_diversity(sources),
                "freshness": self._calculate_freshness(structured_data)
            }
        }
        
        self.update_progress(1.0, "Data collection completed")
        self.logger.info(f"Collected {len(structured_data)} trends from {len(sources)} sources")
        
        return collection_result
    
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
                        current_trend['impact'] = 'high'
                    elif 'medium' in impact_text:
                        current_trend['impact'] = 'medium'
                    else:
                        current_trend['impact'] = 'low'
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
                trend.setdefault('impact', 'medium')
                trend.setdefault('time_horizon', 'medium_term')
                
                validated_trends.append(trend)
        
        self.logger.debug(f"Parsed {len(validated_trends)} trends from LLM response")
        return validated_trends
    
    async def _query_data_sources(self, query: str, sources: List[str]) -> Dict[str, Any]:
        """Simulate querying various data sources (placeholder for real implementations)"""
        # In a real implementation, this would:
        # 1. Query news APIs (NewsAPI, Bing News, etc.)
        # 2. Scrape relevant websites
        # 3. Access social media APIs
        # 4. Query research databases
        # 5. Check patent databases
        
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
        llm_trends: List[Dict[str, Any]], 
        source_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Structure and enrich collected data with source information"""
        
        structured_trends = []
        
        for i, trend in enumerate(llm_trends):
            # Add unique identifier
            trend['id'] = f"trend_{i+1}_{datetime.now().strftime('%Y%m%d')}"
            
            # Add source enrichment
            trend['sources'] = list(source_data.keys())
            
            # Add timestamps
            trend['collected_at'] = datetime.now().isoformat()
            
            # Calculate enhanced confidence based on source corroboration
            base_confidence = trend.get('confidence', 0.6)
            source_boost = min(0.2, len(source_data) * 0.05)  # Max 0.2 boost
            trend['confidence'] = min(0.95, base_confidence + source_boost)
            
            # Add validation flags
            trend['validation'] = {
                'title_length_ok': len(trend['title']) >= 5,
                'has_description': bool(trend.get('description', '').strip()),
                'has_keywords': len(trend.get('keywords', [])) > 0,
                'category_valid': trend.get('category') in ['technology', 'business', 'social', 'environmental']
            }
            
            structured_trends.append(trend)
        
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
    