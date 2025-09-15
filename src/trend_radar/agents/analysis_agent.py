"""
Analysis Agent - Responsible for analyzing trends and scoring their impact, confidence, and timeline.
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple
from ..agents.base_agent import MCPAgent
from ..models.mcp_message import AgentCapability


class AnalysisAgent(MCPAgent):
    """Agent specialized in trend analysis, scoring, and risk assessment"""
    
    def __init__(self, llm_base_url: str = "http://localhost:11434", model_name: str = "gpt-oss:20b", **kwargs):
        super().__init__("analysis_agent", llm_base_url=llm_base_url, model_name=model_name)
        
        # Analysis frameworks and methodologies
        self.impact_factors = [
            "market_size_potential",
            "disruption_level", 
            "adoption_barriers",
            "competitive_landscape",
            "regulatory_environment",
            "technological_readiness"
        ]
        
        self.confidence_factors = [
            "data_quality",
            "source_credibility",
            "expert_consensus", 
            "historical_precedent",
            "signal_strength"
        ]
    
    def _define_capabilities(self) -> List[AgentCapability]:
        """Define analysis capabilities"""
        return [
            AgentCapability(
                name="trend_impact_analysis",
                description="Assess potential impact of trends across multiple dimensions",
                input_types=["trend_data", "market_context"],
                output_types=["impact_scores", "risk_assessment"],
                confidence_level=0.85
            ),
            AgentCapability(
                name="confidence_scoring",
                description="Calculate confidence levels based on data quality and sources",
                input_types=["trend_data", "source_metadata"],
                output_types=["confidence_scores", "uncertainty_bounds"],
                confidence_level=0.8
            ),
            AgentCapability(
                name="timeline_prediction", 
                description="Predict trend development timelines and milestones",
                input_types=["trend_data", "adoption_curves"],
                output_types=["time_horizons", "milestone_predictions"],
                confidence_level=0.75
            ),
            AgentCapability(
                name="risk_opportunity_analysis",
                description="Identify risks and opportunities associated with trends",
                input_types=["trend_data", "business_context"],
                output_types=["risk_factors", "opportunity_matrix"],
                confidence_level=0.8
            ),
            AgentCapability(
                name="competitive_intelligence",
                description="Analyze competitive implications and market positioning",
                input_types=["trend_data", "competitor_data"],
                output_types=["competitive_analysis", "positioning_recommendations"],
                confidence_level=0.7
            )
        ]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main trend analysis pipeline
        
        Args:
            task: Dictionary containing raw trends and analysis parameters
            
        Returns:
            Dictionary with analyzed and scored trends
        """
        raw_trends = task.get("raw_trends", [])
        analysis_depth = task.get("depth", "standard")  # light, standard, deep
        focus_areas = task.get("focus_areas", ["impact", "confidence", "timeline"])
        
        self.logger.info(f"Starting analysis of {len(raw_trends)} trends")
        self.update_progress(0.1, "Initializing trend analysis")
        
        analyzed_trends = []
        
        for i, trend_data in enumerate(raw_trends):
            self.logger.debug(f"Analyzing trend: {trend_data.get('title', 'Unknown')}")
            
            # Core analysis
            analyzed_trend = await self._analyze_single_trend(
                trend_data, 
                analysis_depth, 
                focus_areas
            )
            
            analyzed_trends.append(analyzed_trend)
            
            # Update progress
            progress = 0.1 + (0.7 * (i + 1) / len(raw_trends))
            self.update_progress(progress, f"Analyzed {i+1}/{len(raw_trends)} trends")
        
        # Cross-trend analysis
        self.update_progress(0.85, "Performing cross-trend analysis")
        cross_analysis = await self._perform_cross_trend_analysis(analyzed_trends)
        
        # Generate analysis summary
        self.update_progress(0.95, "Generating analysis summary")
        analysis_summary = self._generate_analysis_summary(analyzed_trends, cross_analysis)
        
        # Compile results
        analysis_result = {
            "analyzed_trends": analyzed_trends,
            "cross_analysis": cross_analysis,
            "analysis_summary": analysis_summary,
            "analysis_metadata": {
                "trends_analyzed": len(analyzed_trends),
                "analysis_depth": analysis_depth,
                "focus_areas": focus_areas,
                "timestamp": datetime.now().isoformat(),
                "analysis_version": "1.0"
            },
            "quality_metrics": {
                "avg_confidence": self._calculate_avg_confidence(analyzed_trends),
                "high_impact_count": self._count_high_impact_trends(analyzed_trends),
                "analysis_completeness": self._calculate_analysis_completeness(analyzed_trends)
            }
        }
        
        self.update_progress(1.0, "Analysis completed")
        self.logger.info(f"Analysis completed for {len(analyzed_trends)} trends")
        
        return analysis_result
    
    async def _analyze_single_trend(
        self, 
        trend_data: Dict[str, Any], 
        depth: str, 
        focus_areas: List[str]
    ) -> Dict[str, Any]:
        """Analyze a single trend across multiple dimensions"""
        
        analyzed_trend = trend_data.copy()
        
        # Impact Analysis
        if "impact" in focus_areas:
            impact_analysis = await self._analyze_impact(trend_data, depth)
            analyzed_trend.update(impact_analysis)
        
        # Confidence Scoring
        if "confidence" in focus_areas:
            confidence_analysis = await self._analyze_confidence(trend_data, depth)
            analyzed_trend.update(confidence_analysis)
        
        # Timeline Prediction
        if "timeline" in focus_areas:
            timeline_analysis = await self._analyze_timeline(trend_data, depth)
            analyzed_trend.update(timeline_analysis)
        
        # Risk/Opportunity Analysis
        if depth in ["standard", "deep"]:
            risk_opportunity = await self._analyze_risks_opportunities(trend_data, depth)
            analyzed_trend.update(risk_opportunity)
        
        # Add analysis metadata
        analyzed_trend["analysis_metadata"] = {
            "analyzed_at": datetime.now().isoformat(),
            "analysis_depth": depth,
            "focus_areas": focus_areas,
            "analyzer_version": "1.0"
        }
        
        return analyzed_trend
    
    async def _analyze_impact(self, trend_data: Dict[str, Any], depth: str) -> Dict[str, Any]:
        """Analyze trend impact using LLM-assisted evaluation"""
        
        system_prompt = """You are a strategic analyst specializing in trend impact assessment. 
        Evaluate trends based on their potential to create significant change across markets, 
        society, and technology landscapes."""
        
        prompt = f"""
        Analyze the impact of this trend:
        
        Title: {trend_data.get('title', 'Unknown')}
        Description: {trend_data.get('description', 'No description')}
        Category: {trend_data.get('category', 'technology')}
        Keywords: {', '.join(trend_data.get('keywords', []))}
        
        Provide analysis in this format:
        IMPACT_LEVEL: [LOW/MEDIUM/HIGH/CRITICAL]
        MARKET_IMPACT: [0-10 scale and brief explanation]
        SOCIAL_IMPACT: [0-10 scale and brief explanation]  
        TECH_IMPACT: [0-10 scale and brief explanation]
        DISRUPTION_POTENTIAL: [LOW/MEDIUM/HIGH]
        AFFECTED_SECTORS: [list key sectors that would be impacted]
        IMPACT_TIMEFRAME: [when significant impact would be felt]
        
        Provide reasoning for your assessments.
        """
        
        response = await self.query_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=400
        )
        
        return self._parse_impact_analysis(response, trend_data)
    
    def _parse_impact_analysis(self, analysis_text: str, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM impact analysis response"""
        lines = analysis_text.lower().split('\n')
        impact_data = {}
        
        # Default values
        impact_mapping = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        impact_data["impact_score"] = 2  # Default medium
        impact_data["market_impact"] = 5.0
        impact_data["social_impact"] = 5.0
        impact_data["tech_impact"] = 5.0
        impact_data["disruption_potential"] = "medium"
        impact_data["affected_sectors"] = []
        
        for line in lines:
            line = line.strip()
            
            if "impact_level:" in line:
                level = line.split(':')[1].strip()
                if level in impact_mapping:
                    impact_data["impact_score"] = impact_mapping[level]
                    impact_data["impact"] = level
            
            elif "market_impact:" in line:
                try:
                    score_part = line.split(':')[1].strip()
                    score = float(score_part.split()[0])
                    impact_data["market_impact"] = max(0, min(10, score))
                except (ValueError, IndexError):
                    pass
            
            elif "social_impact:" in line:
                try:
                    score_part = line.split(':')[1].strip()
                    score = float(score_part.split()[0])
                    impact_data["social_impact"] = max(0, min(10, score))
                except (ValueError, IndexError):
                    pass
            
            elif "tech_impact:" in line:
                try:
                    score_part = line.split(':')[1].strip()
                    score = float(score_part.split()[0])
                    impact_data["tech_impact"] = max(0, min(10, score))
                except (ValueError, IndexError):
                    pass
            
            elif "disruption_potential:" in line:
                potential = line.split(':')[1].strip()
                if potential in ["low", "medium", "high"]:
                    impact_data["disruption_potential"] = potential
            
            elif "affected_sectors:" in line:
                sectors_text = line.split(':')[1].strip()
                sectors = [s.strip() for s in sectors_text.split(',') if s.strip()]
                impact_data["affected_sectors"] = sectors[:5]  # Limit to 5 sectors
        
        # Calculate composite impact score
        composite_score = (
            impact_data["market_impact"] * 0.4 +
            impact_data["social_impact"] * 0.3 +
            impact_data["tech_impact"] * 0.3
        ) / 10.0  # Normalize to 0-1
        
        impact_data["composite_impact_score"] = round(composite_score, 2)
        
        return impact_data
    
    async def _analyze_confidence(self, trend_data: Dict[str, Any], depth: str) -> Dict[str, Any]:
        """Analyze confidence level in trend prediction"""
        
        # Base confidence from data collector
        base_confidence = trend_data.get('confidence', 0.5)
        
        # Confidence factors analysis
        data_quality_score = self._assess_data_quality(trend_data)
        source_credibility_score = self._assess_source_credibility(trend_data)
        signal_strength_score = self._assess_signal_strength(trend_data)
        
        # LLM-assisted confidence analysis for complex cases
        if depth in ["standard", "deep"]:
            llm_confidence = await self._llm_confidence_analysis(trend_data)
        else:
            llm_confidence = {"expert_assessment": base_confidence}
        
        # Weighted confidence calculation
        weights = {
            "base_confidence": 0.3,
            "data_quality": 0.25,
            "source_credibility": 0.2,
            "signal_strength": 0.15,
            "expert_assessment": 0.1
        }
        
        final_confidence = (
            base_confidence * weights["base_confidence"] +
            data_quality_score * weights["data_quality"] +
            source_credibility_score * weights["source_credibility"] +
            signal_strength_score * weights["signal_strength"] +
            llm_confidence.get("expert_assessment", base_confidence) * weights["expert_assessment"]
        )
        
        return {
            "confidence_score": round(max(0.1, min(0.95, final_confidence)), 2),
            "confidence_factors": {
                "data_quality": round(data_quality_score, 2),
                "source_credibility": round(source_credibility_score, 2),
                "signal_strength": round(signal_strength_score, 2),
                "expert_assessment": round(llm_confidence.get("expert_assessment", base_confidence), 2)
            },
            "confidence_level": self._categorize_confidence(final_confidence)
        }
    
    def _assess_data_quality(self, trend_data: Dict[str, Any]) -> float:
        """Assess data quality based on completeness and structure"""
        score = 0.5  # Base score
        
        # Check completeness
        if trend_data.get('title') and len(trend_data['title']) > 5:
            score += 0.1
        if trend_data.get('description') and len(trend_data['description']) > 20:
            score += 0.1
        if trend_data.get('keywords') and len(trend_data['keywords']) > 2:
            score += 0.1
        if trend_data.get('category') in ['technology', 'business', 'social', 'environmental']:
            score += 0.1
        
        # Check validation flags if available
        validation = trend_data.get('validation', {})
        valid_fields = sum(1 for v in validation.values() if v)
        score += (valid_fields / len(validation)) * 0.2 if validation else 0
        
        return min(1.0, score)
    
    def _assess_source_credibility(self, trend_data: Dict[str, Any]) -> float:
        """Assess credibility of data sources"""
        sources = trend_data.get('sources', [])
        
        if not sources:
            return 0.3
        
        credibility_scores = {
            'research_papers': 0.9,
            'tech_blogs': 0.7,
            'news_apis': 0.8,
            'social_media_trends': 0.5,
            'patent_databases': 0.85,
            'startup_databases': 0.6
        }
        
        total_credibility = sum(credibility_scores.get(source, 0.5) for source in sources)
        return min(1.0, total_credibility / len(sources))
    
    def _assess_signal_strength(self, trend_data: Dict[str, Any]) -> float:
        """Assess strength of trend signals"""
        # This would analyze various signals in a real implementation
        # For now, use keyword count and description quality as proxies
        
        keywords = trend_data.get('keywords', [])
        description = trend_data.get('description', '')
        
        signal_score = 0.4  # Base score
        
        # Keyword diversity
        if len(keywords) > 3:
            signal_score += 0.2
        elif len(keywords) > 1:
            signal_score += 0.1
        
        # Description quality (length as proxy for detail)
        if len(description) > 100:
            signal_score += 0.2
        elif len(description) > 50:
            signal_score += 0.1
        
        # Category specificity
        if trend_data.get('category') != 'technology':  # Non-tech trends might be less obvious
            signal_score += 0.1
        
        return min(1.0, signal_score)
    
    async def _llm_confidence_analysis(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to assess confidence in trend prediction"""
        
        prompt = f"""
        Assess the confidence level for this trend prediction:
        
        Title: {trend_data.get('title', 'Unknown')}
        Description: {trend_data.get('description', 'No description')}
        Category: {trend_data.get('category', 'technology')}
        
        Consider:
        1. How well-established are the underlying technologies/concepts?
        2. Are there clear market drivers supporting this trend?
        3. What are the adoption barriers?
        4. How predictable is the timeline?
        
        Provide:
        CONFIDENCE_SCORE: [0.0-1.0]
        REASONING: [brief explanation]
        UNCERTAINTY_FACTORS: [key sources of uncertainty]
        """
        
        response = await self.query_llm(
            prompt=prompt,
            temperature=0.3,
            max_tokens=300
        )
        
        # Parse confidence score
        expert_score = 0.5  # Default
        for line in response.lower().split('\n'):
            if 'confidence_score:' in line:
                try:
                    score_text = line.split(':')[1].strip()
                    expert_score = float(score_text)
                    break
                except (ValueError, IndexError):
                    pass
        
        return {"expert_assessment": max(0.1, min(0.95, expert_score))}
    
    def _categorize_confidence(self, confidence_score: float) -> str:
        """Categorize confidence score into levels"""
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        elif confidence_score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    async def _analyze_timeline(self, trend_data: Dict[str, Any], depth: str) -> Dict[str, Any]:
        """Analyze trend timeline and maturity"""
        
        system_prompt = """You are a technology adoption and trend timeline expert. 
        Analyze trends to predict their development timeline and key milestones."""
        
        prompt = f"""
        Analyze the timeline for this trend:
        
        Title: {trend_data.get('title', 'Unknown')}
        Description: {trend_data.get('description', 'No description')}
        Category: {trend_data.get('category', 'technology')}
        
        Provide:
        TIME_HORIZON: [EMERGING/SHORT_TERM/MEDIUM_TERM/LONG_TERM]
        MATURITY_STAGE: [CONCEPT/EARLY_DEVELOPMENT/PILOT/SCALING/MAINSTREAM]
        KEY_MILESTONES: [list 2-3 key milestones with timeframes]
        ADOPTION_CURVE: [SLOW/MODERATE/RAPID/EXPLOSIVE]
        TIMELINE_CONFIDENCE: [0.0-1.0]
        """
        
        response = await self.query_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=350
        )
        
        return self._parse_timeline_analysis(response)
    
    def _parse_timeline_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse timeline analysis from LLM response"""
        lines = analysis_text.lower().split('\n')
        timeline_data = {}
        
        # Defaults
        timeline_data["time_horizon"] = "medium_term"
        timeline_data["maturity_stage"] = "early_development"
        timeline_data["adoption_curve"] = "moderate"
        timeline_data["timeline_confidence"] = 0.6
        timeline_data["key_milestones"] = []
        
        for line in lines:
            line = line.strip()
            
            if "time_horizon:" in line:
                horizon = line.split(':')[1].strip()
                valid_horizons = ["emerging", "short_term", "medium_term", "long_term"]
                if horizon in valid_horizons:
                    timeline_data["time_horizon"] = horizon
            
            elif "maturity_stage:" in line:
                stage = line.split(':')[1].strip()
                valid_stages = ["concept", "early_development", "pilot", "scaling", "mainstream"]
                if stage in valid_stages:
                    timeline_data["maturity_stage"] = stage
            
            elif "adoption_curve:" in line:
                curve = line.split(':')[1].strip()
                valid_curves = ["slow", "moderate", "rapid", "explosive"]
                if curve in valid_curves:
                    timeline_data["adoption_curve"] = curve
            
            elif "timeline_confidence:" in line:
                try:
                    conf = float(line.split(':')[1].strip())
                    timeline_data["timeline_confidence"] = max(0.1, min(0.95, conf))
                except (ValueError, IndexError):
                    pass
            
            elif "key_milestones:" in line:
                # Simple milestone extraction
                milestones_text = line.split(':', 1)[1].strip()
                if milestones_text:
                    timeline_data["key_milestones"] = [milestones_text[:100]]
        
        return timeline_data
    
    async def _analyze_risks_opportunities(self, trend_data: Dict[str, Any], depth: str) -> Dict[str, Any]:
        """Analyze risks and opportunities associated with the trend"""
        
        prompt = f"""
        Analyze risks and opportunities for this trend:
        
        Title: {trend_data.get('title', 'Unknown')}
        Description: {trend_data.get('description', 'No description')}
        
        Identify:
        TOP_RISKS: [list 2-3 key risks with brief descriptions]
        TOP_OPPORTUNITIES: [list 2-3 key opportunities with brief descriptions]
        RISK_LEVEL: [LOW/MEDIUM/HIGH]
        OPPORTUNITY_LEVEL: [LOW/MEDIUM/HIGH]
        MITIGATION_STRATEGIES: [key strategies to address risks]
        """
        
        response = await self.query_llm(
            prompt=prompt,
            temperature=0.5,
            max_tokens=400
        )
        
        return self._parse_risk_opportunity_analysis(response)
    
    def _parse_risk_opportunity_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse risk and opportunity analysis"""
        lines = analysis_text.split('\n')
        risk_opp_data = {
            "risks": [],
            "opportunities": [],
            "risk_level": "medium",
            "opportunity_level": "medium",
            "mitigation_strategies": []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            lower_line = line.lower()
            
            if "top_risks:" in lower_line:
                current_section = "risks"
            elif "top_opportunities:" in lower_line:
                current_section = "opportunities" 
            elif "risk_level:" in lower_line:
                level = line.split(':')[1].strip().lower()
                if level in ["low", "medium", "high"]:
                    risk_opp_data["risk_level"] = level
            elif "opportunity_level:" in lower_line:
                level = line.split(':')[1].strip().lower()
                if level in ["low", "medium", "high"]:
                    risk_opp_data["opportunity_level"] = level
            elif current_section and line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                item = line.lstrip('-•*123. ').strip()
                if item and current_section in risk_opp_data:
                    risk_opp_data[current_section].append(item)
        
        return risk_opp_data
    
    async def _perform_cross_trend_analysis(self, analyzed_trends: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform cross-trend analysis to identify patterns and correlations"""
        
        if len(analyzed_trends) < 2:
            return {"message": "Insufficient trends for cross-analysis"}
        
        # Category distribution
        category_dist = {}
        impact_dist = {}
        timeline_dist = {}
        
        for trend in analyzed_trends:
            # Categories
            cat = trend.get('category', 'unknown')
            category_dist[cat] = category_dist.get(cat, 0) + 1
            
            # Impact levels
            impact = trend.get('impact', 'medium')
            impact_dist[impact] = impact_dist.get(impact, 0) + 1
            
            # Time horizons
            horizon = trend.get('time_horizon', 'medium_term')
            timeline_dist[horizon] = timeline_dist.get(horizon, 0) + 1
        
        # Identify trend clusters and patterns
        high_impact_trends = [t for t in analyzed_trends if t.get('impact_score', 2) >= 3]
        emerging_trends = [t for t in analyzed_trends if t.get('time_horizon') == 'emerging']
        high_confidence_trends = [t for t in analyzed_trends if t.get('confidence_score', 0.5) >= 0.7]
        
        return {
            "distributions": {
                "categories": category_dist,
                "impact_levels": impact_dist,
                "time_horizons": timeline_dist
            },
            "trend_clusters": {
                "high_impact": len(high_impact_trends),
                "emerging": len(emerging_trends),
                "high_confidence": len(high_confidence_trends)
            },
            "patterns": {
                "dominant_category": max(category_dist.items(), key=lambda x: x[1])[0] if category_dist else "none",
                "avg_confidence": sum(t.get('confidence_score', 0.5) for t in analyzed_trends) / len(analyzed_trends),
                "risk_opportunity_ratio": self._calculate_risk_opportunity_ratio(analyzed_trends)
            }
        }
    
    def _calculate_risk_opportunity_ratio(self, trends: List[Dict[str, Any]]) -> float:
        """Calculate ratio of opportunities to risks across all trends"""
        total_risks = 0
        total_opportunities = 0
        
        for trend in trends:
            total_risks += len(trend.get('risks', []))
            total_opportunities += len(trend.get('opportunities', []))
        
        if total_risks == 0:
            return float('inf') if total_opportunities > 0 else 1.0
        
        return round(total_opportunities / total_risks, 2)
    
    def _generate_analysis_summary(
        self, 
        analyzed_trends: List[Dict[str, Any]], 
        cross_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis summary"""
        
        if not analyzed_trends:
            return {"message": "No trends to summarize"}
        
        # Key metrics
        avg_impact = sum(t.get('impact_score', 2) for t in analyzed_trends) / len(analyzed_trends)
        avg_confidence = sum(t.get('confidence_score', 0.5) for t in analyzed_trends) / len(analyzed_trends)
        
        # Top trends by different criteria
        top_impact_trends = sorted(
            analyzed_trends, 
            key=lambda x: x.get('impact_score', 2), 
            reverse=True
        )[:3]
        
        top_confidence_trends = sorted(
            analyzed_trends,
            key=lambda x: x.get('confidence_score', 0.5),
            reverse=True
        )[:3]
        
        return {
            "overview": {
                "total_trends": len(analyzed_trends),
                "avg_impact_score": round(avg_impact, 2),
                "avg_confidence_score": round(avg_confidence, 2),
                "analysis_quality": "high" if avg_confidence > 0.7 else "medium" if avg_confidence > 0.5 else "low"
            },
            "top_trends": {
                "highest_impact": [{"title": t.get('title', 'Unknown'), "impact_score": t.get('impact_score', 2)} for t in top_impact_trends],
                "highest_confidence": [{"title": t.get('title', 'Unknown'), "confidence_score": t.get('confidence_score', 0.5)} for t in top_confidence_trends]
            },
            "recommendations": self._generate_recommendations(analyzed_trends, cross_analysis)
        }
    
    def _generate_recommendations(
        self, 
        analyzed_trends: List[Dict[str, Any]], 
        cross_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate strategic recommendations based on analysis"""
        recommendations = []
        
        patterns = cross_analysis.get('patterns', {})
        clusters = cross_analysis.get('trend_clusters', {})
        
        # Confidence-based recommendations
        avg_confidence = patterns.get('avg_confidence', 0.5)
        if avg_confidence < 0.6:
            recommendations.append("Consider gathering additional data sources to improve trend confidence levels")
        
        # Impact-based recommendations
        high_impact_count = clusters.get('high_impact', 0)
        if high_impact_count > 0:
            recommendations.append(f"Prioritize monitoring and planning for {high_impact_count} high-impact trends")
        
        # Timeline-based recommendations
        emerging_count = clusters.get('emerging', 0)
        if emerging_count > 0:
            recommendations.append(f"Establish early warning systems for {emerging_count} emerging trends")
        
        # Category-based recommendations
        dominant_cat = patterns.get('dominant_category', 'none')
        if dominant_cat != 'none':
            recommendations.append(f"Focus strategic planning on {dominant_cat} sector trends")
        
        return recommendations
    
    def _calculate_avg_confidence(self, trends: List[Dict[str, Any]]) -> float:
        """Calculate average confidence score"""
        if not trends:
            return 0.0
        return round(sum(t.get('confidence_score', 0.5) for t in trends) / len(trends), 2)
    
    def _count_high_impact_trends(self, trends: List[Dict[str, Any]]) -> int:
        """Count trends with high impact scores"""
        return sum(1 for t in trends if t.get('impact_score', 2) >= 3)
    
    def _calculate_analysis_completeness(self, trends: List[Dict[str, Any]]) -> float:
        """Calculate completeness of analysis"""
        if not trends:
            return 0.0
        
        required_fields = ['impact_score', 'confidence_score', 'time_horizon']
        total_checks = len(trends) * len(required_fields)
        completed_checks = 0
        
        for trend in trends:
            for field in required_fields:
                if field in trend:
                    completed_checks += 1
        
        return round(completed_checks / total_checks, 2) if total_checks > 0 else 0.0
    