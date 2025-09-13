from typing import List, Dict, Any
from datetime import datetime
from trend_radar.agents.base_agent import MCPAgent
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

