#!/usr/bin/env python3
"""
Trend Radar MCP Application
A demonstration of Model Context Protocol with four specialized agents
for comprehensive trend analysis and visualization.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendCategory(Enum):
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"

class TrendImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TrendTimeHorizon(Enum):
    EMERGING = "emerging"      # 0-6 months
    SHORT_TERM = "short_term"  # 6-18 months
    MEDIUM_TERM = "medium_term" # 1-3 years
    LONG_TERM = "long_term"    # 3+ years

@dataclass
class Trend:
    id: str
    title: str
    description: str
    category: TrendCategory
    impact: TrendImpact
    time_horizon: TrendTimeHorizon
    confidence_score: float  # 0.0 to 1.0
    sources: List[str]
    keywords: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['category'] = self.category.value
        data['impact'] = self.impact.value
        data['time_horizon'] = self.time_horizon.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class MCPMessage:
    """Model Context Protocol Message Structure"""
    def __init__(self, agent_id: str, message_type: str, content: Dict[str, Any]):
        self.agent_id = agent_id
        self.message_type = message_type
        self.content = content
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'message_type': self.message_type,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }

class MCPAgent(ABC):
    """Base class for MCP Agents"""
    
    def __init__(self, agent_id: str, model_endpoint: str = "http://localhost:11434/api/generate"):
        self.agent_id = agent_id
        self.model_endpoint = model_endpoint
        self.context_window = []
        self.capabilities = []
        
    async def send_message(self, message: MCPMessage, orchestrator):
        """Send message through MCP to orchestrator"""
        logger.info(f"{self.agent_id} sending message: {message.message_type}")
        await orchestrator.receive_message(message)
        
    async def query_model(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Query the local LLM model (gpt-oss:20b via Ollama)"""
        payload = {
            "model": "gpt-oss:20b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500
            }
        }
        
        if context:
            payload["context"] = context
            
        try:
            response = requests.post(
                self.model_endpoint,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Model query failed: {e}")
            return f"Error querying model: {str(e)}"
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process assigned task"""
        pass

class DataCollectorAgent(MCPAgent):
    """Agent responsible for collecting trend data from various sources"""
    
    def __init__(self):
        super().__init__("data_collector")
        self.capabilities = ["web_scraping", "api_integration", "data_extraction"]
        self.sources = [
            "tech_blogs", "news_apis", "social_media", "research_papers"
        ]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Collect raw trend data"""
        query = task.get("query", "emerging technology trends")
        
        prompt = f"""
        Act as a data collection specialist. Based on the query '{query}', 
        identify and describe 3-5 emerging trends with the following information:
        - Trend title
        - Brief description
        - Category (technology/business/social/environmental)
        - Key indicators or signals
        - Potential sources where this trend was observed
        
        Format your response as structured data that can be parsed.
        """
        
        response = await self.query_model(prompt)
        
        # Simulate collected data (in real implementation, this would query actual sources)
        collected_data = {
            "raw_trends": self._parse_trend_data(response),
            "collection_timestamp": datetime.now().isoformat(),
            "sources_queried": self.sources,
            "query": query
        }
        
        return collected_data
    
    def _parse_trend_data(self, model_response: str) -> List[Dict]:
        """Parse model response into structured trend data"""
        # This is a simplified parser - in production, use more robust parsing
        trends = []
        lines = model_response.split('\n')
        current_trend = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Trend:') or line.startswith('Title:'):
                if current_trend:
                    trends.append(current_trend)
                current_trend = {'title': line.split(':', 1)[1].strip()}
            elif line.startswith('Description:'):
                current_trend['description'] = line.split(':', 1)[1].strip()
            elif line.startswith('Category:'):
                current_trend['category'] = line.split(':', 1)[1].strip().lower()
        
        if current_trend:
            trends.append(current_trend)
            
        return trends

class AnalysisAgent(MCPAgent):
    """Agent responsible for analyzing and scoring trends"""
    
    def __init__(self):
        super().__init__("analysis_agent")
        self.capabilities = ["trend_analysis", "impact_assessment", "confidence_scoring"]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends for impact, confidence, and time horizon"""
        raw_trends = task.get("raw_trends", [])
        
        analyzed_trends = []
        
        for trend_data in raw_trends:
            prompt = f"""
            Analyze this trend for a trend radar:
            Title: {trend_data.get('title', 'Unknown')}
            Description: {trend_data.get('description', 'No description')}
            Category: {trend_data.get('category', 'technology')}
            
            Provide analysis in this format:
            Impact Level: [LOW/MEDIUM/HIGH/CRITICAL]
            Time Horizon: [EMERGING/SHORT_TERM/MEDIUM_TERM/LONG_TERM]
            Confidence Score: [0.0-1.0]
            Risk Factors: [list key risks]
            Opportunities: [list key opportunities]
            Keywords: [comma-separated relevant keywords]
            """
            
            analysis = await self.query_model(prompt)
            analyzed_trend = self._parse_analysis(trend_data, analysis)
            analyzed_trends.append(analyzed_trend)
        
        return {
            "analyzed_trends": analyzed_trends,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_method": "llm_assisted"
        }
    
    def _parse_analysis(self, trend_data: Dict, analysis_text: str) -> Dict:
        """Parse analysis results into structured format"""
        analyzed = trend_data.copy()
        lines = analysis_text.lower().split('\n')
        
        for line in lines:
            if 'impact level:' in line:
                impact = line.split(':')[1].strip()
                analyzed['impact'] = impact if impact in ['low', 'medium', 'high', 'critical'] else 'medium'
            elif 'time horizon:' in line:
                horizon = line.split(':')[1].strip()
                analyzed['time_horizon'] = horizon if horizon in ['emerging', 'short_term', 'medium_term', 'long_term'] else 'medium_term'
            elif 'confidence score:' in line:
                try:
                    score = float(line.split(':')[1].strip())
                    analyzed['confidence_score'] = max(0.0, min(1.0, score))
                except:
                    analyzed['confidence_score'] = 0.5
            elif 'keywords:' in line:
                keywords = line.split(':')[1].strip().split(',')
                analyzed['keywords'] = [k.strip() for k in keywords if k.strip()]
        
        return analyzed

class VisualizationAgent(MCPAgent):
    """Agent responsible for creating trend radar visualizations"""
    
    def __init__(self):
        super().__init__("visualization_agent")
        self.capabilities = ["radar_generation", "data_visualization", "report_creation"]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization data and configurations"""
        analyzed_trends = task.get("analyzed_trends", [])
        
        # Create radar chart data
        radar_data = self._create_radar_data(analyzed_trends)
        
        # Generate visualization config
        viz_config = {
            "chart_type": "radar",
            "dimensions": {
                "impact": ["low", "medium", "high", "critical"],
                "time_horizon": ["emerging", "short_term", "medium_term", "long_term"]
            },
            "color_scheme": {
                "technology": "#FF6B6B",
                "business": "#4ECDC4",
                "social": "#45B7D1",
                "environmental": "#96CEB4"
            }
        }
        
        # Generate summary statistics
        stats = self._calculate_statistics(analyzed_trends)
        
        return {
            "radar_data": radar_data,
            "visualization_config": viz_config,
            "statistics": stats,
            "generated_at": datetime.now().isoformat()
        }
    
    def _create_radar_data(self, trends: List[Dict]) -> List[Dict]:
        """Convert analyzed trends to radar chart format"""
        radar_points = []
        
        impact_values = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        horizon_values = {"emerging": 1, "short_term": 2, "medium_term": 3, "long_term": 4}
        
        for i, trend in enumerate(trends):
            point = {
                "id": f"trend_{i}",
                "title": trend.get("title", "Unknown Trend"),
                "category": trend.get("category", "technology"),
                "x": horizon_values.get(trend.get("time_horizon", "medium_term"), 3),
                "y": impact_values.get(trend.get("impact", "medium"), 2),
                "size": trend.get("confidence_score", 0.5) * 20 + 5,
                "description": trend.get("description", ""),
                "keywords": trend.get("keywords", [])
            }
            radar_points.append(point)
        
        return radar_points
    
    def _calculate_statistics(self, trends: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for the trend dataset"""
        if not trends:
            return {}
        
        category_counts = {}
        impact_counts = {}
        horizon_counts = {}
        total_confidence = 0
        
        for trend in trends:
            # Category distribution
            cat = trend.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Impact distribution
            impact = trend.get("impact", "medium")
            impact_counts[impact] = impact_counts.get(impact, 0) + 1
            
            # Time horizon distribution
            horizon = trend.get("time_horizon", "medium_term")
            horizon_counts[horizon] = horizon_counts.get(horizon, 0) + 1
            
            # Average confidence
            total_confidence += trend.get("confidence_score", 0.5)
        
        return {
            "total_trends": len(trends),
            "category_distribution": category_counts,
            "impact_distribution": impact_counts,
            "time_horizon_distribution": horizon_counts,
            "average_confidence": round(total_confidence / len(trends), 2)
        }

class ReportingAgent(MCPAgent):
    """Agent responsible for generating reports and insights"""
    
    def __init__(self):
        super().__init__("reporting_agent")
        self.capabilities = ["report_generation", "insight_extraction", "recommendation_creation"]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trend report"""
        radar_data = task.get("radar_data", [])
        statistics = task.get("statistics", {})
        
        # Generate executive summary
        summary_prompt = f"""
        Based on the following trend analysis statistics, write an executive summary:
        - Total trends analyzed: {statistics.get('total_trends', 0)}
        - Category distribution: {statistics.get('category_distribution', {})}
        - Impact distribution: {statistics.get('impact_distribution', {})}
        - Average confidence: {statistics.get('average_confidence', 0)}
        
        Provide key insights, recommendations, and strategic implications.
        """
        
        executive_summary = await self.query_model(summary_prompt)
        
        # Generate trend insights
        insights = await self._generate_insights(radar_data)
        
        # Create recommendations
        recommendations = await self._generate_recommendations(radar_data, statistics)
        
        report = {
            "executive_summary": executive_summary,
            "key_insights": insights,
            "strategic_recommendations": recommendations,
            "trend_details": radar_data,
            "statistics": statistics,
            "report_generated_at": datetime.now().isoformat(),
            "report_version": "1.0"
        }
        
        return report
    
    async def _generate_insights(self, radar_data: List[Dict]) -> List[str]:
        """Extract key insights from trend data"""
        high_impact_trends = [t for t in radar_data if t.get('y', 0) >= 3]
        emerging_trends = [t for t in radar_data if t.get('x', 0) == 1]
        
        insights = []
        
        if high_impact_trends:
            insights.append(f"Identified {len(high_impact_trends)} high-impact trends requiring immediate attention")
        
        if emerging_trends:
            insights.append(f"Detected {len(emerging_trends)} emerging trends in early stages")
        
        # Category analysis
        categories = {}
        for trend in radar_data:
            cat = trend.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        dominant_category = max(categories.items(), key=lambda x: x[1]) if categories else None
        if dominant_category:
            insights.append(f"Technology category '{dominant_category[0]}' shows highest activity with {dominant_category[1]} trends")
        
        return insights
    
    async def _generate_recommendations(self, radar_data: List[Dict], statistics: Dict) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        avg_confidence = statistics.get('average_confidence', 0)
        if avg_confidence < 0.6:
            recommendations.append("Consider additional data sources to improve trend confidence scores")
        
        high_impact_count = len([t for t in radar_data if t.get('y', 0) >= 3])
        if high_impact_count > 3:
            recommendations.append("Prioritize resource allocation for high-impact trend monitoring")
        
        emerging_count = len([t for t in radar_data if t.get('x', 0) == 1])
        if emerging_count > 0:
            recommendations.append("Establish early warning systems for emerging trends")
        
        return recommendations

class TrendRadarOrchestrator:
    """Main orchestrator that coordinates all agents using MCP"""
    
    def __init__(self):
        self.agents = {
            "data_collector": DataCollectorAgent(),
            "analysis_agent": AnalysisAgent(),
            "visualization_agent": VisualizationAgent(),
            "reporting_agent": ReportingAgent()
        }
        self.message_queue = asyncio.Queue()
        self.results = {}
        
    async def receive_message(self, message: MCPMessage):
        """Receive messages from agents via MCP"""
        await self.message_queue.put(message)
        
    async def orchestrate_trend_analysis(self, query: str = "emerging technology trends") -> Dict[str, Any]:
        """Main orchestration workflow"""
        logger.info("Starting trend radar analysis orchestration")
        
        try:
            # Step 1: Data Collection
            logger.info("Step 1: Initiating data collection")
            collection_task = {"query": query}
            collection_result = await self.agents["data_collector"].process_task(collection_task)
            
            # Step 2: Analysis
            logger.info("Step 2: Analyzing collected trends")
            analysis_task = {"raw_trends": collection_result["raw_trends"]}
            analysis_result = await self.agents["analysis_agent"].process_task(analysis_task)
            
            # Step 3: Visualization
            logger.info("Step 3: Generating visualization data")
            viz_task = {"analyzed_trends": analysis_result["analyzed_trends"]}
            viz_result = await self.agents["visualization_agent"].process_task(viz_task)
            
            # Step 4: Reporting
            logger.info("Step 4: Creating comprehensive report")
            report_task = {
                "radar_data": viz_result["radar_data"],
                "statistics": viz_result["statistics"]
            }
            report_result = await self.agents["reporting_agent"].process_task(report_task)
            
            # Compile final results
            final_result = {
                "trend_radar": {
                    "data": viz_result["radar_data"],
                    "config": viz_result["visualization_config"],
                    "statistics": viz_result["statistics"]
                },
                "report": report_result,
                "metadata": {
                    "query": query,
                    "processing_complete": True,
                    "timestamp": datetime.now().isoformat(),
                    "agents_involved": list(self.agents.keys())
                }
            }
            
            logger.info("Trend radar analysis completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {"error": str(e), "processing_complete": False}

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_radar_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")

async def main():
    """Main application entry point"""
    print("🎯 Trend Radar MCP Application")
    print("=" * 50)
    
    orchestrator = TrendRadarOrchestrator()
    
    # Get user input for trend analysis
    query = input("Enter trend analysis query (or press Enter for default): ").strip()
    if not query:
        query = "emerging technology trends 2025"
    
    print(f"\n🔍 Analyzing trends for: '{query}'")
    print("🤖 Coordinating MCP agents...")
    
    # Run the analysis
    results = await orchestrator.orchestrate_trend_analysis(query)
    
    if results.get("processing_complete"):
        print("\n✅ Analysis Complete!")
        print("\n📊 Trend Radar Summary:")
        print(f"- Total trends identified: {results['trend_radar']['statistics'].get('total_trends', 0)}")
        print(f"- Average confidence: {results['trend_radar']['statistics'].get('average_confidence', 0)}")
        
        print("\n📋 Key Insights:")
        for insight in results['report'].get('key_insights', []):
            print(f"  • {insight}")
        
        print("\n💡 Recommendations:")
        for rec in results['report'].get('strategic_recommendations', []):
            print(f"  • {rec}")
        
        # Save results
        orchestrator.save_results(results)
        print(f"\n💾 Full results saved to JSON file")
        
        print("\n🎯 Trend Radar Data Points:")
        for trend in results['trend_radar']['data']:
            print(f"  • {trend['title']} ({trend['category']}) - Impact: {trend['y']}/4, Timeline: {trend['x']}/4")
    
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    print("Starting Trend Radar MCP Application...")
    print("Note: This application requires Ollama with gpt-oss:20b model running locally")
    print("Install: `ollama pull gpt-oss:20b` and ensure Ollama is running on port 11434\n")
    
    asyncio.run(main())
    