"""
Reporting Agent - Responsible for generating comprehensive reports and strategic insights.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from ..agents.base_agent import MCPAgent
from ..models.mcp_message import AgentCapability

class ReportingAgent(MCPAgent):
    """Agent specialized in generating reports, insights, and strategic recommendations"""
    
    def __init__(self, llm_base_url: str = "http://localhost:11434", model_name: str = "gpt-oss:20b", **kwargs):
        super().__init__("reporting_agent", llm_base_url=llm_base_url, model_name=model_name)
        
        # Report templates and formats
        self.report_templates = {
            "executive": "executive_summary",
            "detailed": "detailed_analysis",
            "strategic": "strategic_recommendations",
            "technical": "technical_deep_dive"
        }
        
        # Insight generation frameworks
        self.insight_frameworks = [
            "trend_convergence",
            "market_disruption_patterns", 
            "adoption_lifecycle_analysis",
            "competitive_landscape_shifts",
            "risk_opportunity_matrix"
        ]
    
    def _define_capabilities(self) -> List[AgentCapability]:
        """Define reporting and insight capabilities"""
        return [
            AgentCapability(
                name="executive_reporting",
                description="Generate executive-level summaries and strategic overviews",
                input_types=["trend_analysis", "visualization_data"],
                output_types=["executive_summary", "key_insights"],
                confidence_level=0.9
            ),
            AgentCapability(
                name="detailed_analysis_reports",
                description="Create comprehensive analytical reports with deep insights", 
                input_types=["analyzed_trends", "cross_analysis", "market_context"],
                output_types=["detailed_report", "analysis_documentation"],
                confidence_level=0.85
            ),
            AgentCapability(
                name="strategic_recommendations",
                description="Generate actionable strategic recommendations",
                input_types=["trend_insights", "business_context", "risk_analysis"],
                output_types=["strategic_plan", "recommendation_matrix"],
                confidence_level=0.8
            ),
            AgentCapability(
                name="insight_extraction",
                description="Extract key insights and patterns from trend data",
                input_types=["trend_data", "statistical_analysis"],
                output_types=["key_insights", "pattern_analysis"], 
                confidence_level=0.75
            ),
            AgentCapability(
                name="competitive_intelligence_reports",
                description="Generate competitive intelligence and market positioning reports",
                input_types=["trend_data", "competitor_analysis", "market_data"],
                output_types=["competitive_report", "positioning_recommendations"],
                confidence_level=0.7
            )
        ]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main reporting pipeline
        
        Args:
            task: Dictionary containing radar data, statistics, and reporting parameters
            
        Returns:
            Dictionary with comprehensive report and insights
        """
        radar_data = task.get("radar_data", [])
        statistics = task.get("statistics", {})
        supporting_charts = task.get("supporting_charts", {})
        report_type = task.get("type", "comprehensive")  # executive, detailed, strategic, comprehensive
        target_audience = task.get("audience", "leadership")  # leadership, technical, board, stakeholders
        
        self.logger.info(f"Starting {report_type} report generation for {target_audience}")
        self.update_progress(0.1, "Initializing report generation")
        
        # Generate executive summary
        self.update_progress(0.3, "Creating executive summary")
        executive_summary = await self._generate_executive_summary(radar_data, statistics, target_audience)
        
        # Extract key insights
        self.update_progress(0.5, "Extracting key insights")
        key_insights = await self._extract_key_insights(radar_data, statistics, supporting_charts)
        
        # Generate strategic recommendations
        self.update_progress(0.7, "Generating strategic recommendations")
        strategic_recommendations = await self._generate_strategic_recommendations(radar_data, statistics, key_insights)
        
        # Create detailed analysis sections
        self.update_progress(0.85, "Creating detailed analysis")
        detailed_analysis = await self._create_detailed_analysis(radar_data, statistics, supporting_charts, report_type)
        
        # Compile comprehensive report
        self.update_progress(0.95, "Compiling final report")
        final_report = self._compile_final_report(
            executive_summary,
            key_insights,
            strategic_recommendations,
            detailed_analysis,
            radar_data,
            statistics,
            report_type,
            target_audience
        )
        
        self.update_progress(1.0, "Report generation completed")
        self.logger.info(f"Generated {report_type} report with {len(key_insights)} insights")
        
        return final_report
    
    async def _generate_executive_summary(
        self, 
        radar_data: List[Dict[str, Any]], 
        statistics: Dict[str, Any],
        target_audience: str
    ) -> str:
        """Generate executive summary using LLM"""
        
        # Prepare context for LLM
        total_trends = len(radar_data)
        avg_confidence = statistics.get("overview", {}).get("average_confidence", 0.5)
        quadrant_analysis = statistics.get("quadrant_analysis", {})
        
        # Get top trends by impact
        high_impact_trends = sorted(
            radar_data, 
            key=lambda x: x.get('y', 2), 
            reverse=True
        )[:5]
        
        system_prompt = f"""You are a strategic analyst writing for {target_audience}. 
        Create a compelling executive summary that highlights the most important findings 
        and their strategic implications. Focus on actionable insights and business impact."""
        
        prompt = f"""
        Based on trend radar analysis, write an executive summary:
        
        ANALYSIS OVERVIEW:
        - Total trends analyzed: {total_trends}
        - Average confidence level: {avg_confidence:.1%}
        - Quadrant distribution: {quadrant_analysis}
        
        TOP HIGH-IMPACT TRENDS:
        {self._format_trends_for_llm(high_impact_trends[:3])}
        
        Write a 200-300 word executive summary covering:
        1. Key findings and overall trend landscape
        2. Most significant opportunities and threats
        3. Strategic implications for the organization
        4. Recommended immediate actions
        
        Use clear, business-focused language appropriate for {target_audience}.
        """
        
        summary = await self.query_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=400
        )
        
        return summary.strip()
    
    def _format_trends_for_llm(self, trends: List[Dict[str, Any]]) -> str:
        """Format trends for LLM consumption"""
        formatted_trends = []
        
        for trend in trends:
            formatted_trends.append(
                f"• {trend.get('title', 'Unknown')} ({trend.get('category', 'unknown')})\n"
                f"  Impact: {trend.get('y', 2)}/4, Confidence: {trend.get('confidence', 0.5):.1%}\n"
                f"  {trend.get('description', '')[:100]}..."
            )
        
        return "\n".join(formatted_trends)
    
    async def _extract_key_insights(
        self,
        radar_data: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        supporting_charts: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract key insights using multiple analysis frameworks"""
        
        insights = []
        
        # Quantitative insights from statistics
        insights.extend(self._extract_quantitative_insights(statistics, radar_data))
        
        # Pattern-based insights
        pattern_insights = await self._extract_pattern_insights(radar_data)
        insights.extend(pattern_insights)
        
        # Comparative insights
        comparative_insights = self._extract_comparative_insights(radar_data, supporting_charts)
        insights.extend(comparative_insights)
        
        # LLM-generated insights for complex patterns
        llm_insights = await self._generate_llm_insights(radar_data, statistics)
        insights.extend(llm_insights)
        
        # Rank and filter insights by importance
        ranked_insights = self._rank_insights(insights)
        
        return ranked_insights[:10]  # Top 10 insights
    
    def _extract_quantitative_insights(
        self, 
        statistics: Dict[str, Any], 
        radar_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract insights from quantitative analysis"""
        
        insights = []
        overview = statistics.get("overview", {})
        distributions = statistics.get("distributions", {})
        quadrant_analysis = statistics.get("quadrant_analysis", {})
        
        # Confidence insights
        avg_confidence = overview.get("average_confidence", 0.5)
        if avg_confidence > 0.8:
            insights.append({
                "type": "confidence",
                "title": "High Confidence in Trend Predictions",
                "description": f"Analysis shows {avg_confidence:.1%} average confidence, indicating strong data quality and reliable predictions.",
                "importance": 0.8,
                "category": "quality"
            })
        elif avg_confidence < 0.5:
            insights.append({
                "type": "confidence",
                "title": "Low Confidence Signals Need Attention", 
                "description": f"Average confidence of {avg_confidence:.1%} suggests need for additional data sources and validation.",
                "importance": 0.9,
                "category": "risk"
            })
        
        # Impact distribution insights
        impact_dist = distributions.get("impact_levels", {})
        high_impact_count = impact_dist.get("high", 0) + impact_dist.get("critical", 0)
        if high_impact_count > len(radar_data) * 0.3:  # More than 30% high impact
            insights.append({
                "type": "impact",
                "title": "High Concentration of High-Impact Trends",
                "description": f"{high_impact_count} trends show high impact potential, requiring prioritized resource allocation.",
                "importance": 0.9,
                "category": "opportunity"
            })
        
        # Quadrant insights
        strategic_bets = quadrant_analysis.get("strategic_bets", 0)
        quick_wins = quadrant_analysis.get("quick_wins", 0)
        
        if quick_wins > strategic_bets * 2:
            insights.append({
                "type": "timing",
                "title": "Near-term Opportunities Dominate",
                "description": f"{quick_wins} quick-win opportunities vs {strategic_bets} strategic bets suggest focus on immediate actions.",
                "importance": 0.7,
                "category": "timing"
            })
        
        return insights
    
    async def _extract_pattern_insights(self, radar_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract pattern-based insights using LLM analysis"""
        
        # Analyze trend clusters and correlations
        categories = {}
        for trend in radar_data:
            cat = trend.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(trend)
        
        pattern_prompt = f"""
        Analyze these trend patterns and identify key insights:
        
        CATEGORY BREAKDOWN:
        {self._format_category_breakdown(categories)}
        
        COORDINATE DISTRIBUTION:
        {self._format_coordinate_distribution(radar_data)}
        
        Identify 2-3 key pattern-based insights such as:
        - Trend convergence or clustering patterns
        - Gaps in the radar that represent opportunities
        - Unusual distributions that suggest market dynamics
        - Cross-category correlations or dependencies
        
        Format each insight as: INSIGHT_TYPE: Title - Description
        """
        
        response = await self.query_llm(
            prompt=pattern_prompt,
            temperature=0.5,
            max_tokens=400
        )
        
        return self._parse_llm_insights(response, "pattern")
    

    def _format_category_breakdown(self, categories: Dict[str, List]) -> str:
        """Format category breakdown for LLM analysis"""
        breakdown = []
        for cat, trends in categories.items():
            avg_impact = sum(t.get('y', 2) for t in trends) / len(trends)
            avg_confidence = sum(t.get('confidence', 0.5) for t in trends) / len(trends)
            breakdown.append(f"• {cat.title()}: {len(trends)} trends, avg impact {avg_impact:.1f}, avg confidence {avg_confidence:.1%}")


    def _format_coordinate_distribution(self, radar_data: List[Dict[str, Any]]) -> str:
            """Format coordinate distribution for analysis"""
            x_coords = [t.get('x', 3) for t in radar_data]
            y_coords = [t.get('y', 2) for t in radar_data]
            
            return f"X (Time): min={min(x_coords)}, max={max(x_coords)}, avg={sum(x_coords)/len(x_coords):.1f}\n" + \
                f"Y (Impact): min={min(y_coords)}, max={max(y_coords)}, avg={sum(y_coords)/len(y_coords):.1f}"
        
    def _extract_comparative_insights(
        self, 
        radar_data: List[Dict[str, Any]], 
        supporting_charts: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract insights through comparative analysis"""
        
        insights = []
        
        # Category comparison
        category_counts = {}
        category_impacts = {}
        
        for trend in radar_data:
            cat = trend.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            if cat not in category_impacts:
                category_impacts[cat] = []
            category_impacts[cat].append(trend.get('y', 2))
        
        # Find dominant category
        if category_counts:
            dominant_cat = max(category_counts.items(), key=lambda x: x[1])
            if dominant_cat[1] > len(radar_data) * 0.4:  # >40% of trends
                insights.append({
                    "type": "category_dominance",
                    "title": f"{dominant_cat[0].title()} Trends Dominate Landscape",
                    "description": f"{dominant_cat[0].title()} represents {dominant_cat[1]}/{len(radar_data)} trends, suggesting sector-specific market dynamics.",
                    "importance": 0.7,
                    "category": "market_dynamics"
                })
        
        # Compare average impacts by category
        if len(category_impacts) > 1:
            cat_avg_impacts = {cat: sum(impacts)/len(impacts) for cat, impacts in category_impacts.items()}
            highest_impact_cat = max(cat_avg_impacts.items(), key=lambda x: x[1])
            lowest_impact_cat = min(cat_avg_impacts.items(), key=lambda x: x[1])
            
            if highest_impact_cat[1] - lowest_impact_cat[1] > 1.0:  # Significant difference
                insights.append({
                    "type": "impact_variance",
                    "title": "Significant Impact Variance Across Categories",
                    "description": f"{highest_impact_cat[0].title()} trends show {highest_impact_cat[1]:.1f} avg impact vs {lowest_impact_cat[0]} at {lowest_impact_cat[1]:.1f}.",
                    "importance": 0.6,
                    "category": "comparative"
                })
        
        return insights
    
    async def _generate_llm_insights(
        self, 
        radar_data: List[Dict[str, Any]], 
        statistics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate complex insights using LLM analysis"""
        
        # Select diverse trends for analysis
        sample_trends = self._select_representative_trends(radar_data)
        
        insight_prompt = f"""
        Analyze this trend radar data for strategic insights:
        
        SAMPLE TRENDS:
        {self._format_trends_for_llm(sample_trends)}
        
        STATISTICAL CONTEXT:
        {json.dumps(statistics, indent=2)[:500]}...
        
        Generate 2-3 strategic insights focusing on:
        1. Market disruption opportunities
        2. Competitive positioning implications
        3. Resource allocation recommendations
        4. Risk mitigation priorities
        
        Each insight should be strategic, actionable, and based on the data patterns.
        Format: CATEGORY: Title - Detailed description with implications
        """
        
        response = await self.query_llm(
            prompt=insight_prompt,
            temperature=0.6,
            max_tokens=500
        )
        
        return self._parse_llm_insights(response, "strategic")
    
    def _select_representative_trends(self, radar_data: List[Dict[str, Any]], count: int = 5) -> List[Dict[str, Any]]:
        """Select diverse, representative trends for LLM analysis"""
        if len(radar_data) <= count:
            return radar_data
        
        # Sort by impact and select from different quadrants
        sorted_by_impact = sorted(radar_data, key=lambda x: x.get('y', 2), reverse=True)
        
        selected = []
        categories_seen = set()
        quadrants_seen = set()
        
        for trend in sorted_by_impact:
            if len(selected) >= count:
                break
            
            category = trend.get('category', 'unknown')
            x, y = trend.get('x', 3), trend.get('y', 2)
            quadrant = f"{x>=2.5}_{y>=2.5}"  # Simple quadrant classification
            
            # Prefer trends from unseen categories/quadrants
            if category not in categories_seen or quadrant not in quadrants_seen:
                selected.append(trend)
                categories_seen.add(category)
                quadrants_seen.add(quadrant)
            elif len(selected) < count - 1:  # Fill remaining slots
                selected.append(trend)
        
        return selected
    
    def _parse_llm_insights(self, response: str, insight_type: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured insights"""
        insights = []
        lines = response.strip().split('\n')
        
        current_insight = None
        importance_map = {"strategic": 0.8, "pattern": 0.7, "comparative": 0.6}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for insight headers (CATEGORY: Title - Description)
            if ':' in line and any(keyword in line.upper() for keyword in ['INSIGHT', 'OPPORTUNITY', 'RISK', 'PATTERN', 'STRATEGIC']):
                if current_insight:
                    insights.append(current_insight)
                
                parts = line.split(':', 1)
                if len(parts) == 2:
                    category = parts[0].strip()
                    title_desc = parts[1].strip()
                    
                    if ' - ' in title_desc:
                        title, description = title_desc.split(' - ', 1)
                    else:
                        title = title_desc
                        description = ""
                    
                    current_insight = {
                        "type": insight_type,
                        "title": title.strip(),
                        "description": description.strip(),
                        "importance": importance_map.get(insight_type, 0.5),
                        "category": category.lower()
                    }
            elif current_insight and line:
                # Add to description if we're building an insight
                if current_insight["description"]:
                    current_insight["description"] += " " + line
                else:
                    current_insight["description"] = line
        
        # Don't forget the last insight
        if current_insight:
            insights.append(current_insight)
        
        return insights
    
    def _rank_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank insights by importance and uniqueness"""
        # Remove duplicates based on title similarity
        unique_insights = []
        seen_titles = set()
        
        for insight in insights:
            title_words = set(insight.get('title', '').lower().split())
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if len(title_words.intersection(seen_words)) > len(title_words) * 0.6:  # 60% word overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_insights.append(insight)
                seen_titles.add(insight.get('title', '').lower())
        
        # Sort by importance
        return sorted(unique_insights, key=lambda x: x.get('importance', 0.5), reverse=True)
    
    async def _generate_strategic_recommendations(
        self,
        radar_data: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        key_insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate actionable strategic recommendations"""
        
        recommendations = []
        
        # Data-driven recommendations
        recommendations.extend(self._generate_data_driven_recommendations(radar_data, statistics))
        
        # Insight-based recommendations
        recommendations.extend(self._generate_insight_based_recommendations(key_insights))
        
        # LLM-generated strategic recommendations
        llm_recommendations = await self._generate_llm_recommendations(radar_data, statistics, key_insights)
        recommendations.extend(llm_recommendations)
        
        # Rank and prioritize recommendations
        prioritized_recommendations = self._prioritize_recommendations(recommendations)
        
        return prioritized_recommendations[:8]  # Top 8 recommendations
    
    def _generate_data_driven_recommendations(
        self, 
        radar_data: List[Dict[str, Any]], 
        statistics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on statistical analysis"""
        
        recommendations = []
        overview = statistics.get("overview", {})
        quadrant_analysis = statistics.get("quadrant_analysis", {})
        
        # Confidence-based recommendations
        avg_confidence = overview.get("average_confidence", 0.5)
        if avg_confidence < 0.6:
            recommendations.append({
                "title": "Enhance Data Collection and Validation",
                "description": "Improve trend confidence through additional data sources and expert validation",
                "priority": "high",
                "timeframe": "immediate",
                "effort": "medium",
                "impact": "foundational"
            })
        
        # Quadrant-based recommendations
        quick_wins = quadrant_analysis.get("quick_wins", 0)
        strategic_bets = quadrant_analysis.get("strategic_bets", 0)
        
        if quick_wins > 0:
            recommendations.append({
                "title": "Execute Quick-Win Initiatives",
                "description": f"Prioritize {quick_wins} high-impact, near-term trends for immediate competitive advantage",
                "priority": "high",
                "timeframe": "0-6 months", 
                "effort": "low-medium",
                "impact": "immediate"
            })
        
        if strategic_bets > 0:
            recommendations.append({
                "title": "Develop Long-term Strategic Capabilities",
                "description": f"Invest in {strategic_bets} high-impact, long-term trends for future market positioning",
                "priority": "medium",
                "timeframe": "1-3 years",
                "effort": "high",
                "impact": "transformational"
            })
        
        return recommendations
    
    def _generate_insight_based_recommendations(self, key_insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on extracted insights"""
        
        recommendations = []
        
        for insight in key_insights[:5]:  # Top 5 insights
            insight_type = insight.get('type', 'general')
            category = insight.get('category', 'general')
            
            if category == 'opportunity':
                recommendations.append({
                    "title": f"Capitalize on {insight.get('title', 'Identified Opportunity')}",
                    "description": f"Leverage insight: {insight.get('description', '')[:100]}...",
                    "priority": "medium",
                    "timeframe": "3-12 months",
                    "effort": "medium",
                    "impact": "strategic"
                })
            
            elif category == 'risk':
                recommendations.append({
                    "title": f"Mitigate Risk: {insight.get('title', 'Identified Risk')}",
                    "description": f"Address risk factor: {insight.get('description', '')[:100]}...",
                    "priority": "high",
                    "timeframe": "immediate",
                    "effort": "medium",
                    "impact": "protective"
                })
        
        return recommendations
    
    async def _generate_llm_recommendations(
        self,
        radar_data: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        key_insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate strategic recommendations using LLM"""
        
        top_trends = sorted(radar_data, key=lambda x: x.get('y', 2), reverse=True)[:5]
        insight_summary = "\n".join([f"• {i.get('title', 'Unknown')}: {i.get('description', '')[:80]}..." for i in key_insights[:3]])
        
        recommendation_prompt = f"""
        Based on this trend analysis, generate 3-4 strategic recommendations:
        
        TOP TRENDS:
        {self._format_trends_for_llm(top_trends[:3])}
        
        KEY INSIGHTS:
        {insight_summary}
        
        Generate actionable recommendations covering:
        1. Technology investment priorities
        2. Market positioning strategies  
        3. Capability building needs
        4. Risk mitigation approaches
        
        For each recommendation, specify:
        - Clear action title
        - Detailed description and rationale
        - Priority level (high/medium/low)
        - Timeframe for implementation
        - Required effort level
        - Expected impact type
        
        Format: PRIORITY: Title - Description [Timeframe: X, Effort: Y, Impact: Z]
        """
        
        response = await self.query_llm(
            prompt=recommendation_prompt,
            temperature=0.5,
            max_tokens=600
        )
        
        return self._parse_llm_recommendations(response)
    
    def _parse_llm_recommendations(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured recommendations"""
        recommendations = []
        lines = response.strip().split('\n')
        
        current_rec = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for recommendation headers
            if any(priority in line.upper() for priority in ['HIGH:', 'MEDIUM:', 'LOW:']):
                if current_rec:
                    recommendations.append(current_rec)
                
                # Parse priority and title
                parts = line.split(':', 1)
                if len(parts) == 2:
                    priority = parts[0].strip().lower()
                    title_desc = parts[1].strip()
                    
                    # Extract title and description
                    if ' - ' in title_desc:
                        title, description = title_desc.split(' - ', 1)
                    else:
                        title = title_desc
                        description = ""
                    
                    # Extract metadata if present [Timeframe: X, Effort: Y, Impact: Z]
                    timeframe = "3-12 months"  # default
                    effort = "medium"  # default
                    impact = "strategic"  # default
                    
                    if '[' in description and ']' in description:
                        metadata_start = description.find('[')
                        metadata_end = description.find(']')
                        metadata = description[metadata_start+1:metadata_end]
                        description = description[:metadata_start].strip()
                        
                        # Parse metadata
                        for item in metadata.split(','):
                            if 'timeframe:' in item.lower():
                                timeframe = item.split(':')[1].strip()
                            elif 'effort:' in item.lower():
                                effort = item.split(':')[1].strip()
                            elif 'impact:' in item.lower():
                                impact = item.split(':')[1].strip()
                    
                    current_rec = {
                        "title": title.strip(),
                        "description": description.strip(),
                        "priority": priority,
                        "timeframe": timeframe,
                        "effort": effort,
                        "impact": impact
                    }
            elif current_rec and line:
                # Add to description
                if current_rec["description"]:
                    current_rec["description"] += " " + line
                else:
                    current_rec["description"] = line
        
        # Don't forget the last recommendation
        if current_rec:
            recommendations.append(current_rec)
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize recommendations by impact and urgency"""
        
        priority_scores = {"high": 3, "medium": 2, "low": 1}
        impact_scores = {"transformational": 4, "strategic": 3, "immediate": 2, "foundational": 2, "protective": 1}
        
        for rec in recommendations:
            priority_score = priority_scores.get(rec.get('priority', 'medium'), 2)
            impact_score = impact_scores.get(rec.get('impact', 'strategic'), 3)
            rec['_score'] = priority_score * impact_score
        
        return sorted(recommendations, key=lambda x: x.get('_score', 0), reverse=True)
    
    async def _create_detailed_analysis(
        self,
        radar_data: List[Dict[str, Any]],
        statistics: Dict[str, Any], 
        supporting_charts: Dict[str, Any],
        report_type: str
    ) -> Dict[str, Any]:
        """Create detailed analysis sections"""
        
        detailed_analysis = {}
        
        # Trend analysis by category
        detailed_analysis["category_analysis"] = self._analyze_by_category(radar_data)
        
        # Quadrant analysis
        detailed_analysis["quadrant_analysis"] = self._analyze_by_quadrant(radar_data, statistics)
        
        # Risk and opportunity matrix
        detailed_analysis["risk_opportunity_matrix"] = self._create_risk_opportunity_matrix(radar_data)
        
        # Timeline analysis
        detailed_analysis["timeline_analysis"] = self._analyze_timeline_distribution(radar_data)
        
        # Confidence analysis
        detailed_analysis["confidence_analysis"] = self._analyze_confidence_patterns(radar_data)
        
        if report_type in ["detailed", "comprehensive"]:
            # Individual trend deep-dives for top trends
            detailed_analysis["trend_deep_dives"] = await self._create_trend_deep_dives(radar_data[:5])
        
        return detailed_analysis
    
    def _analyze_by_category(self, radar_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends by category"""
        
        categories = {}
        
        for trend in radar_data:
            cat = trend.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = {
                    "count": 0,
                    "trends": [],
                    "avg_impact": 0,
                    "avg_confidence": 0,
                    "impact_range": {"min": 4, "max": 1}
                }
            
            categories[cat]["count"] += 1
            categories[cat]["trends"].append(trend.get('title', 'Unknown'))
            
            impact = trend.get('y', 2)
            confidence = trend.get('confidence', 0.5)
            
            categories[cat]["avg_impact"] += impact
            categories[cat]["avg_confidence"] += confidence
            categories[cat]["impact_range"]["min"] = min(categories[cat]["impact_range"]["min"], impact)
            categories[cat]["impact_range"]["max"] = max(categories[cat]["impact_range"]["max"], impact)
        
        # Calculate averages
        for cat_data in categories.values():
            if cat_data["count"] > 0:
                cat_data["avg_impact"] = round(cat_data["avg_impact"] / cat_data["count"], 2)
                cat_data["avg_confidence"] = round(cat_data["avg_confidence"] / cat_data["count"], 2)
        
        return categories
    
    def _analyze_by_quadrant(self, radar_data: List[Dict[str, Any]], statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Detailed quadrant analysis"""
        
        quadrant_data = {}
        quadrant_analysis = statistics.get("quadrant_analysis", {})
        
        quadrants = {
            "quick_wins": {"x_max": 2, "y_min": 3, "name": "Quick Wins"},
            "strategic_bets": {"x_min": 3, "y_min": 3, "name": "Strategic Bets"},
            "tactical_moves": {"x_max": 2, "y_max": 2, "name": "Tactical Moves"},
            "background_noise": {"x_min": 3, "y_max": 2, "name": "Background Noise"}
        }
        
        for quad_key, quad_def in quadrants.items():
            trends_in_quadrant = []
            
            for trend in radar_data:
                x, y = trend.get('x', 3), trend.get('y', 2)
                
                # Check if trend belongs to this quadrant
                if quad_key == "quick_wins" and x <= 2 and y >= 3:
                    trends_in_quadrant.append(trend)
                elif quad_key == "strategic_bets" and x >= 3 and y >= 3:
                    trends_in_quadrant.append(trend)
                elif quad_key == "tactical_moves" and x <= 2 and y <= 2:
                    trends_in_quadrant.append(trend)
                elif quad_key == "background_noise" and x >= 3 and y <= 2:
                    trends_in_quadrant.append(trend)
            
            quadrant_data[quad_key] = {
                "name": quad_def["name"],
                "count": len(trends_in_quadrant),
                "trends": [t.get('title', 'Unknown') for t in trends_in_quadrant],
                "avg_confidence": sum(t.get('confidence', 0.5) for t in trends_in_quadrant) / len(trends_in_quadrant) if trends_in_quadrant else 0,
                "recommended_action": self._get_quadrant_recommendation(quad_key)
            }
        
        return quadrant_data
    
    def _get_quadrant_recommendation(self, quadrant: str) -> str:
        """Get recommended action for each quadrant"""
        recommendations = {
            "quick_wins": "Prioritize immediate implementation and resource allocation",
            "strategic_bets": "Develop long-term capabilities and strategic partnerships",
            "tactical_moves": "Monitor for potential quick improvements or combinations",
            "background_noise": "Maintain awareness but avoid significant investment"
        }
        return recommendations.get(quadrant, "Monitor and evaluate")
    
    def _create_risk_opportunity_matrix(self, radar_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create risk and opportunity analysis matrix"""
        
        risks = []
        opportunities = []
        
        for trend in radar_data:
            trend_risks = trend.get('risks', [])
            trend_opportunities = trend.get('opportunities', [])
            
            for risk in trend_risks:
                risks.append({
                    "trend": trend.get('title', 'Unknown'),
                    "risk": risk,
                    "impact": trend.get('y', 2),
                    "probability": trend.get('confidence', 0.5)
                })
            
            for opportunity in trend_opportunities:
                opportunities.append({
                    "trend": trend.get('title', 'Unknown'),
                    "opportunity": opportunity,
                    "impact": trend.get('y', 2),
                    "probability": trend.get('confidence', 0.5)
                })
        
        return {
            "risks": sorted(risks, key=lambda x: x['impact'] * x['probability'], reverse=True),
            "opportunities": sorted(opportunities, key=lambda x: x['impact'] * x['probability'], reverse=True),
            "risk_opportunity_ratio": len(opportunities) / len(risks) if risks else float('inf')
        }
    
    def _analyze_timeline_distribution(self, radar_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends by timeline distribution"""
        
        timeline_buckets = {
            "immediate": {"range": "0-6 months", "trends": []},
            "short_term": {"range": "6-18 months", "trends": []},
            "medium_term": {"range": "1-3 years", "trends": []},
            "long_term": {"range": "3+ years", "trends": []}
        }
        
        horizon_mapping = {
            "emerging": "immediate",
            "short_term": "short_term", 
            "medium_term": "medium_term",
            "long_term": "long_term"
        }
        
        for trend in radar_data:
            horizon = trend.get('time_horizon', 'medium_term')
            bucket = horizon_mapping.get(horizon, 'medium_term')
            timeline_buckets[bucket]["trends"].append({
                "title": trend.get('title', 'Unknown'),
                "impact": trend.get('y', 2),
                "confidence": trend.get('confidence', 0.5)
            })
        
        # Calculate timeline statistics
        for bucket_data in timeline_buckets.values():
            trends = bucket_data["trends"]
            if trends:
                bucket_data["count"] = len(trends)
                bucket_data["avg_impact"] = sum(t["impact"] for t in trends) / len(trends)
                bucket_data["avg_confidence"] = sum(t["confidence"] for t in trends) / len(trends)
            else:
                bucket_data["count"] = 0
                bucket_data["avg_impact"] = 0
                bucket_data["avg_confidence"] = 0
        
        return timeline_buckets
    
    def _analyze_confidence_patterns(self, radar_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence patterns across trends"""
        
        confidence_scores = [trend.get('confidence', 0.5) for trend in radar_data]
        
        if not confidence_scores:
            return {}
        
        return {
            "average": sum(confidence_scores) / len(confidence_scores),
            "median": sorted(confidence_scores)[len(confidence_scores) // 2],
            "min": min(confidence_scores),
            "max": max(confidence_scores),
            "high_confidence_count": sum(1 for c in confidence_scores if c >= 0.7),
            "low_confidence_count": sum(1 for c in confidence_scores if c < 0.5),
            "confidence_by_category": self._confidence_by_category(radar_data)
        }
    
    def _confidence_by_category(self, radar_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average confidence by category"""
        
        category_confidence = {}
        
        for trend in radar_data:
            cat = trend.get('category', 'unknown')
            confidence = trend.get('confidence', 0.5)
            
            if cat not in category_confidence:
                category_confidence[cat] = []
            category_confidence[cat].append(confidence)
        
        return {
            cat: sum(confidences) / len(confidences) 
            for cat, confidences in category_confidence.items()
        }
    
    async def _create_trend_deep_dives(self, top_trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed analysis for top trends"""
        
        deep_dives = []
        
        for trend in top_trends:
            deep_dive_prompt = f"""
            Create a detailed analysis for this trend:
            
            Title: {trend.get('title', 'Unknown')}
            Description: {trend.get('description', 'No description')}
            Category: {trend.get('category', 'technology')}
            Impact: {trend.get('y', 2)}/4
            Confidence: {trend.get('confidence', 0.5):.1%}
            Time Horizon: {trend.get('time_horizon_label', 'Medium-term')}
            
            Provide detailed analysis covering:
            1. Market dynamics and drivers
            2. Key stakeholders and players
            3. Technology readiness and barriers
            4. Adoption timeline and milestones
            5. Strategic implications and recommendations
            
            Keep analysis focused and actionable (150-200 words).
            """
            
            analysis = await self.query_llm(
                prompt=deep_dive_prompt,
                temperature=0.4,
                max_tokens=300
            )
            
            deep_dives.append({
                "trend_title": trend.get('title', 'Unknown'),
                "trend_id": trend.get('id', 'unknown'),
                "detailed_analysis": analysis.strip(),
                "key_metrics": {
                    "impact_score": trend.get('y', 2),
                    "confidence_score": trend.get('confidence', 0.5),
                    "time_horizon": trend.get('time_horizon_label', 'Medium-term'),
                    "category": trend.get('category', 'technology')
                }
            })
        
        return deep_dives
    
    def _compile_final_report(
        self,
        executive_summary: str,
        key_insights: List[Dict[str, Any]],
        strategic_recommendations: List[Dict[str, Any]],
        detailed_analysis: Dict[str, Any],
        radar_data: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        report_type: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """Compile all components into final report structure"""
        
        report = {
            "report_metadata": {
                "title": f"Trend Radar Analysis Report - {datetime.now().strftime('%B %Y')}",
                "report_type": report_type,
                "target_audience": target_audience,
                "generated_at": datetime.now().isoformat(),
                "trends_analyzed": len(radar_data),
                "report_version": "1.0",
                "validity_period": "3-6 months"
            },
            "executive_summary": executive_summary,
            "key_insights": key_insights,
            "strategic_recommendations": strategic_recommendations,
            "detailed_analysis": detailed_analysis,
            "appendices": {
                "trend_data": radar_data,
                "statistical_summary": statistics,
                "methodology": self._create_methodology_section(),
                "data_sources": self._create_data_sources_section(),
                "glossary": self._create_glossary()
            }
        }
        
        return report
    

    def _create_methodology_section(self) -> Dict[str, Any]:
        """Create methodology documentation"""
        return {
            "data_collection": "Multi-source aggregation using LLM-assisted trend identification and structured analysis",
            "analysis_framework": "Impact-confidence-timeline matrix with quadrant-based strategic categorization",
            "confidence_calculation": "Weighted scoring based on data quality, source credibility, and expert assessment",
            "validation_process": "Cross-validation through multiple analysis frameworks and pattern recognition",
            "limitations": "Analysis based on available data sources and model capabilities at time of generation"
        }
    
    def _create_data_sources_section(self) -> Dict[str, str]:
        """Create data sources documentation"""
        return {
            "primary_sources": "LLM-generated trend analysis based on training data",
            "secondary_sources": "Simulated industry reports, research publications, and market data",
            "expert_input": "AI-assisted expert analysis and validation",
            "update_frequency": "Analysis reflects point-in-time assessment, recommend quarterly updates"
        }
    
    def _create_glossary(self) -> Dict[str, str]:
        """Create glossary of terms used in the report"""
        return {
            "Impact Level": "Potential magnitude of change (1=Low to 4=Critical)",
            "Confidence Score": "Reliability of trend prediction (0.0-1.0 scale)",
            "Time Horizon": "Expected timeline for trend materialization",
            "Quick Wins": "High-impact, near-term opportunities",
            "Strategic Bets": "High-impact, long-term investments",
            "Tactical Moves": "Low-impact, near-term actions",
            "Background Noise": "Low-impact, long-term monitoring items"
        }

