"""
Visualization Agent - Responsible for creating trend radar visualizations and data representations.
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple
import json
from ..agents.base_agent import MCPAgent
from ..models.mcp_message import AgentCapability
from ..models.trend import RadarPoint


class VisualizationAgent(MCPAgent):
    """Agent specialized in creating trend radar visualizations and charts"""
    
    def __init__(self, llm_base_url: str = "http://localhost:11434", model_name: str = "gpt-oss:20b", **kwargs):
        super().__init__("visualization_agent", llm_base_url=llm_base_url, model_name=model_name)
        
        # Visualization configuration
        self.color_schemes = {
            "technology": "#FF6B6B",
            "business": "#4ECDC4", 
            "social": "#45B7D1",
            "environmental": "#96CEB4"
        }
        
        self.chart_dimensions = {
            "impact": {"low": 1, "medium": 2, "high": 3, "critical": 4},
            "time_horizon": {"emerging": 1, "short_term": 2, "medium_term": 3, "long_term": 4}
        }
    
    def _define_capabilities(self) -> List[AgentCapability]:
        """Define visualization capabilities"""
        return [
            AgentCapability(
                name="radar_chart_generation",
                description="Generate trend radar chart data and configurations",
                input_types=["analyzed_trends", "visualization_params"],
                output_types=["radar_data", "chart_config"],
                confidence_level=0.9
            ),
            AgentCapability(
                name="data_visualization",
                description="Create various chart types and visual representations",
                input_types=["trend_data", "chart_type"],
                output_types=["chart_data", "visualization_config"],
                confidence_level=0.85
            ),
            AgentCapability(
                name="interactive_dashboard",
                description="Generate interactive dashboard configurations",
                input_types=["multi_dataset", "dashboard_params"],
                output_types=["dashboard_config", "widget_definitions"],
                confidence_level=0.8
            ),
            AgentCapability(
                name="report_visualization",
                description="Create visual elements for reports and presentations",
                input_types=["report_data", "format_requirements"],
                output_types=["visual_elements", "layout_configs"],
                confidence_level=0.75
            ),
            AgentCapability(
                name="statistical_charts",
                description="Generate statistical analysis visualizations",
                input_types=["statistics", "analysis_results"],
                output_types=["statistical_charts", "trend_graphs"],
                confidence_level=0.8
            )
        ]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main visualization processing pipeline
        
        Args:
            task: Dictionary containing analyzed trends and visualization parameters
            
        Returns:
            Dictionary with visualization data, configurations, and statistics
        """
        analyzed_trends = task.get("analyzed_trends", [])
        viz_type = task.get("type", "radar")  # radar, dashboard, report
        custom_config = task.get("config", {})
        
        self.logger.info(f"Starting visualization generation for {len(analyzed_trends)} trends")
        self.update_progress(0.1, "Initializing visualization pipeline")
        
        # Generate radar chart data
        self.update_progress(0.3, "Creating radar chart data")
        radar_data = await self._create_radar_data(analyzed_trends)
        
        # Generate visualization configuration
        self.update_progress(0.5, "Generating visualization configuration")
        viz_config = await self._create_visualization_config(analyzed_trends, custom_config)
        
        # Generate supporting charts and statistics
        self.update_progress(0.7, "Creating supporting visualizations")
        supporting_charts = await self._create_supporting_charts(analyzed_trends)
        
        # Calculate visualization statistics
        self.update_progress(0.9, "Calculating visualization statistics")
        viz_statistics = self._calculate_visualization_statistics(analyzed_trends, radar_data)
        
        # Compile visualization results
        visualization_result = {
            "radar_data": radar_data,
            "visualization_config": viz_config,
            "supporting_charts": supporting_charts,
            "statistics": viz_statistics,
            "metadata": {
                "trends_visualized": len(analyzed_trends),
                "visualization_type": viz_type,
                "generated_at": datetime.now().isoformat(),
                "chart_dimensions": self.chart_dimensions,
                "color_scheme": self.color_schemes
            }
        }
        
        self.update_progress(1.0, "Visualization generation completed")
        self.logger.info(f"Generated visualization data for {len(analyzed_trends)} trends")
        
        return visualization_result
    
    async def _create_radar_data(self, analyzed_trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert analyzed trends to radar chart format"""
        radar_points = []
        
        for i, trend in enumerate(analyzed_trends):
            # Map impact and time horizon to coordinates
            impact_score = trend.get('impact_score', 2)
            time_horizon = trend.get('time_horizon', 'medium_term')
            
            # Convert to radar coordinates
            x_coordinate = self.chart_dimensions["time_horizon"].get(time_horizon, 3)
            y_coordinate = impact_score
            
            # Calculate point size based on confidence
            confidence = trend.get('confidence_score', 0.5)
            point_size = max(5, min(30, confidence * 25 + 5))
            
            # Create radar point
            radar_point = RadarPoint(
                id=trend.get('id', f'trend_{i}'),
                title=trend.get('title', 'Unknown Trend'),
                category=trend.get('category', 'technology'),
                x=x_coordinate,
                y=y_coordinate, 
                size=point_size,
                description=trend.get('description', ''),
                keywords=trend.get('keywords', [])
            )
            
            # Add additional visualization metadata
            radar_point_dict = radar_point.to_dict()
            radar_point_dict.update({
                'confidence': confidence,
                'impact_level': trend.get('impact', 'medium'),
                'time_horizon_label': time_horizon.replace('_', ' ').title(),
                'color': self.color_schemes.get(trend.get('category', 'technology'), '#888888'),
                'risks': trend.get('risks', []),
                'opportunities': trend.get('opportunities', []),
                'maturity_stage': trend.get('maturity_stage', 'unknown')
            })
            
            radar_points.append(radar_point_dict)
        
        self.logger.debug(f"Created {len(radar_points)} radar points")
        return radar_points
    
    async def _create_visualization_config(
        self, 
        analyzed_trends: List[Dict[str, Any]], 
        custom_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive visualization configuration"""
        
        # Base configuration
        config = {
            "chart_type": "radar",
            "dimensions": {
                "width": custom_config.get("width", 800),
                "height": custom_config.get("height", 600),
                "margin": {"top": 50, "right": 50, "bottom": 50, "left": 50}
            },
            "axes": {
                "x_axis": {
                    "label": "Time Horizon",
                    "scale": [1, 4],
                    "ticks": [
                        {"value": 1, "label": "Emerging (0-6m)"},
                        {"value": 2, "label": "Short-term (6-18m)"},
                        {"value": 3, "label": "Medium-term (1-3y)"},
                        {"value": 4, "label": "Long-term (3y+)"}
                    ]
                },
                "y_axis": {
                    "label": "Impact Level", 
                    "scale": [1, 4],
                    "ticks": [
                        {"value": 1, "label": "Low"},
                        {"value": 2, "label": "Medium"},
                        {"value": 3, "label": "High"},
                        {"value": 4, "label": "Critical"}
                    ]
                }
            },
            "colors": self.color_schemes,
            "legend": {
                "position": "right",
                "categories": list(set(t.get('category', 'technology') for t in analyzed_trends))
            },
            "interactivity": {
                "hover_enabled": True,
                "click_enabled": True,
                "zoom_enabled": custom_config.get("zoom_enabled", True),
                "filter_enabled": True
            },
            "tooltips": {
                "enabled": True,
                "fields": ["title", "description", "confidence", "impact_level", "time_horizon_label"]
            }
        }
        
        # Add quadrant labels and styling
        config["quadrants"] = await self._define_quadrants()
        
        # Add responsive configuration
        config["responsive"] = {
            "breakpoints": {
                "mobile": {"width": 400, "height": 300},
                "tablet": {"width": 600, "height": 450},
                "desktop": {"width": 800, "height": 600}
            }
        }
        
        return config
    
    async def _define_quadrants(self) -> List[Dict[str, Any]]:
        """Define radar chart quadrants with intelligent labeling"""
        
        # Use LLM to generate contextual quadrant labels
        prompt = """
        Create descriptive labels for a trend radar chart with these quadrants:
        1. High Impact, Long-term (top-right): Trends with major impact but distant timeline
        2. High Impact, Short-term (top-left): Trends with major impact and near-term timeline  
        3. Low Impact, Long-term (bottom-right): Trends with limited impact and distant timeline
        4. Low Impact, Short-term (bottom-left): Trends with limited impact and near-term timeline
        
        Provide creative, business-relevant labels for each quadrant (2-3 words each).
        Format: QUADRANT_NAME: description
        """
        
        response = await self.query_llm(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
        
        # Parse or use defaults
        quadrants = [
            {
                "name": "Strategic Bets",
                "position": {"x_min": 3, "x_max": 4, "y_min": 3, "y_max": 4},
                "description": "High impact, long-term trends requiring strategic investment",
                "color": "rgba(255, 0, 0, 0.1)"
            },
            {
                "name": "Quick Wins", 
                "position": {"x_min": 1, "x_max": 2, "y_min": 3, "y_max": 4},
                "description": "High impact, near-term opportunities for immediate action",
                "color": "rgba(0, 255, 0, 0.1)"
            },
            {
                "name": "Background Noise",
                "position": {"x_min": 3, "x_max": 4, "y_min": 1, "y_max": 2}, 
                "description": "Low impact, long-term trends to monitor passively",
                "color": "rgba(128, 128, 128, 0.1)"
            },
            {
                "name": "Tactical Moves",
                "position": {"x_min": 1, "x_max": 2, "y_min": 1, "y_max": 2},
                "description": "Low impact, near-term trends for tactical consideration",
                "color": "rgba(0, 0, 255, 0.1)"
            }
        ]
        
        return quadrants
    
    async def _create_supporting_charts(self, analyzed_trends: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate supporting charts and visualizations"""
        
        supporting_charts = {}
        
        # Category distribution pie chart
        category_counts = {}
        for trend in analyzed_trends:
            cat = trend.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        supporting_charts['category_distribution'] = {
            "type": "pie",
            "title": "Trends by Category",
            "data": [
                {"label": cat, "value": count, "color": self.color_schemes.get(cat, '#888888')}
                for cat, count in category_counts.items()
            ]
        }
        
        # Impact level distribution bar chart
        impact_counts = {}
        for trend in analyzed_trends:
            impact = trend.get('impact', 'medium')
            impact_counts[impact] = impact_counts.get(impact, 0) + 1
        
        supporting_charts['impact_distribution'] = {
            "type": "bar",
            "title": "Trends by Impact Level",
            "data": [
                {"label": impact.title(), "value": count}
                for impact, count in sorted(impact_counts.items())
            ]
        }
        
        # Time horizon distribution
        horizon_counts = {}
        for trend in analyzed_trends:
            horizon = trend.get('time_horizon', 'medium_term')
            horizon_counts[horizon] = horizon_counts.get(horizon, 0) + 1
        
        supporting_charts['timeline_distribution'] = {
            "type": "horizontal_bar",
            "title": "Trends by Time Horizon", 
            "data": [
                {"label": horizon.replace('_', ' ').title(), "value": count}
                for horizon, count in horizon_counts.items()
            ]
        }
        
        # Confidence vs Impact scatter plot
        confidence_impact_data = []
        for trend in analyzed_trends:
            confidence_impact_data.append({
                "x": trend.get('confidence_score', 0.5),
                "y": trend.get('impact_score', 2),
                "label": trend.get('title', 'Unknown'),
                "category": trend.get('category', 'technology')
            })
        
        supporting_charts['confidence_vs_impact'] = {
            "type": "scatter",
            "title": "Confidence vs Impact Analysis",
            "data": confidence_impact_data,
            "axes": {
                "x": {"label": "Confidence Score", "min": 0, "max": 1},
                "y": {"label": "Impact Score", "min": 1, "max": 4}
            }
        }
        
        # Trend maturity timeline
        maturity_data = {}
        for trend in analyzed_trends:
            maturity = trend.get('maturity_stage', 'unknown')
            if maturity not in maturity_data:
                maturity_data[maturity] = []
            maturity_data[maturity].append({
                "title": trend.get('title', 'Unknown'),
                "confidence": trend.get('confidence_score', 0.5)
            })
        
        supporting_charts['maturity_timeline'] = {
            "type": "timeline",
            "title": "Trend Maturity Stages",
            "data": [
                {"stage": stage, "trends": trends}
                for stage, trends in maturity_data.items()
            ]
        }
        
        return supporting_charts
    
    def _calculate_visualization_statistics(
        self, 
        analyzed_trends: List[Dict[str, Any]], 
        radar_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics for visualization quality and coverage"""
        
        if not analyzed_trends:
            return {}
        
        # Basic statistics
        total_trends = len(analyzed_trends)
        avg_confidence = sum(t.get('confidence_score', 0.5) for t in analyzed_trends) / total_trends
        avg_impact = sum(t.get('impact_score', 2) for t in analyzed_trends) / total_trends
        
        # Distribution statistics
        category_dist = {}
        impact_dist = {}
        horizon_dist = {}
        confidence_ranges = {"low": 0, "medium": 0, "high": 0}
        
        for trend in analyzed_trends:
            # Categories
            cat = trend.get('category', 'unknown')
            category_dist[cat] = category_dist.get(cat, 0) + 1
            
            # Impact
            impact = trend.get('impact', 'medium')
            impact_dist[impact] = impact_dist.get(impact, 0) + 1
            
            # Time horizons
            horizon = trend.get('time_horizon', 'medium_term')
            horizon_dist[horizon] = horizon_dist.get(horizon, 0) + 1
            
            # Confidence ranges
            conf = trend.get('confidence_score', 0.5)
            if conf >= 0.7:
                confidence_ranges["high"] += 1
            elif conf >= 0.4:
                confidence_ranges["medium"] += 1
            else:
                confidence_ranges["low"] += 1
        
        # Quadrant analysis
        quadrant_counts = self._analyze_quadrant_distribution(radar_data)
        
        # Visualization quality metrics
        quality_metrics = {
            "data_completeness": self._calculate_data_completeness(analyzed_trends),
            "visual_balance": self._calculate_visual_balance(radar_data),
            "color_distribution": self._calculate_color_distribution(analyzed_trends),
            "readability_score": self._calculate_readability_score(analyzed_trends)
        }
        
        return {
            "overview": {
                "total_trends": total_trends,
                "average_confidence": round(avg_confidence, 2),
                "average_impact": round(avg_impact, 2),
                "visualization_quality": "high" if avg_confidence > 0.7 else "medium" if avg_confidence > 0.5 else "low"
            },
            "distributions": {
                "categories": category_dist,
                "impact_levels": impact_dist,
                "time_horizons": horizon_dist,
                "confidence_ranges": confidence_ranges
            },
            "quadrant_analysis": quadrant_counts,
            "quality_metrics": quality_metrics,
            "recommendations": self._generate_visualization_recommendations(analyzed_trends, quality_metrics)
        }
    
    def _analyze_quadrant_distribution(self, radar_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how trends are distributed across radar quadrants"""
        
        quadrants = {
            "strategic_bets": 0,    # High impact, long-term (x >= 3, y >= 3)
            "quick_wins": 0,        # High impact, short-term (x <= 2, y >= 3)
            "background_noise": 0,  # Low impact, long-term (x >= 3, y <= 2)
            "tactical_moves": 0     # Low impact, short-term (x <= 2, y <= 2)
        }
        
        for point in radar_data:
            x, y = point.get('x', 3), point.get('y', 2)
            
            if x >= 3 and y >= 3:
                quadrants["strategic_bets"] += 1
            elif x <= 2 and y >= 3:
                quadrants["quick_wins"] += 1
            elif x >= 3 and y <= 2:
                quadrants["background_noise"] += 1
            else:
                quadrants["tactical_moves"] += 1
        
        return quadrants
    
    def _calculate_data_completeness(self, trends: List[Dict[str, Any]]) -> float:
        """Calculate completeness of data for visualization"""
        if not trends:
            return 0.0
        
        required_fields = ['title', 'description', 'category', 'impact_score', 'confidence_score', 'time_horizon']
        total_fields = len(trends) * len(required_fields)
        complete_fields = 0
        
        for trend in trends:
            for field in required_fields:
                if field in trend and trend[field] is not None:
                    if isinstance(trend[field], str) and trend[field].strip():
                        complete_fields += 1
                    elif not isinstance(trend[field], str):
                        complete_fields += 1
        
        return round(complete_fields / total_fields, 2) if total_fields > 0 else 0.0
    
    def _calculate_visual_balance(self, radar_data: List[Dict[str, Any]]) -> float:
        """Calculate how balanced the visual distribution is"""
        if not radar_data:
            return 0.0
        
        # Calculate variance in x and y coordinates
        x_coords = [point.get('x', 3) for point in radar_data]
        y_coords = [point.get('y', 2) for point in radar_data]
        
        x_variance = sum((x - sum(x_coords)/len(x_coords))**2 for x in x_coords) / len(x_coords)
        y_variance = sum((y - sum(y_coords)/len(y_coords))**2 for y in y_coords) / len(y_coords)
        
        # Better balance = lower variance (inverted and normalized)
        max_variance = 1.25  # Theoretical max for 1-4 scale
        balance_score = 1 - min(1, (x_variance + y_variance) / (2 * max_variance))
        
        return round(balance_score, 2)
    
    def _calculate_color_distribution(self, trends: List[Dict[str, Any]]) -> float:
        """Calculate how well distributed the colors/categories are"""
        if not trends:
            return 0.0
        
        categories = [t.get('category', 'technology') for t in trends]
        unique_categories = set(categories)
        
        if len(unique_categories) == 1:
            return 0.5  # All same category
        
        # Calculate Shannon diversity index for color distribution
        category_counts = {cat: categories.count(cat) for cat in unique_categories}
        total = len(categories)
        
        shannon_index = -sum((count/total) * (count/total).bit_length() for count in category_counts.values())
        max_shannon = (len(unique_categories)).bit_length()  # Max possible diversity
        
        return round(shannon_index / max_shannon, 2) if max_shannon > 0 else 0.0
    
    def _calculate_readability_score(self, trends: List[Dict[str, Any]]) -> float:
        """Calculate readability score based on title lengths and description quality"""
        if not trends:
            return 0.0
        
        readability_score = 0.0
        
        for trend in trends:
            title = trend.get('title', '')
            description = trend.get('description', '')
            
            # Title length (optimal: 3-8 words)
            title_words = len(title.split())
            if 3 <= title_words <= 8:
                readability_score += 0.3
            elif title_words > 0:
                readability_score += 0.1
            
            # Description quality (should be substantial but not too long)
            desc_length = len(description)
            if 50 <= desc_length <= 200:
                readability_score += 0.7
            elif desc_length > 10:
                readability_score += 0.3
        
        return round(readability_score / len(trends), 2)
    
    def _generate_visualization_recommendations(
        self, 
        trends: List[Dict[str, Any]], 
        quality_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving visualization"""
        recommendations = []
        
        # Data completeness
        completeness = quality_metrics.get('data_completeness', 0.0)
        if completeness < 0.8:
            recommendations.append("Improve data completeness for better visualization quality")
        
        # Visual balance
        balance = quality_metrics.get('visual_balance', 0.0)
        if balance < 0.6:
            recommendations.append("Consider adjusting trend distribution for better visual balance")
        
        # Color distribution
        color_dist = quality_metrics.get('color_distribution', 0.0)
        if color_dist < 0.5:
            recommendations.append("Diversify trend categories for better visual distinction")
        
        # Readability
        readability = quality_metrics.get('readability_score', 0.0)
        if readability < 0.7:
            recommendations.append("Optimize trend titles and descriptions for better readability")
        
        # Trend count
        if len(trends) > 20:
            recommendations.append("Consider filtering or grouping trends for clearer visualization")
        elif len(trends) < 5:
            recommendations.append("Gather more trend data for comprehensive radar coverage")
        
        return recommendations
    
    async def create_export_formats(self, visualization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create data in various export formats"""
        
        radar_data = visualization_data.get('radar_data', [])
        
        export_formats = {
            "json": {
                "format": "json",
                "data": visualization_data,
                "mime_type": "application/json"
            },
            "csv": {
                "format": "csv", 
                "data": self._convert_to_csv_format(radar_data),
                "mime_type": "text/csv"
            },
            "svg_config": {
                "format": "svg_config",
                "data": await self._create_svg_config(visualization_data),
                "mime_type": "image/svg+xml"
            }
        }
        
        return export_formats
    
    def _convert_to_csv_format(self, radar_data: List[Dict[str, Any]]) -> str:
        """Convert radar data to CSV format"""
        if not radar_data:
            return ""
        
        # Define CSV headers
        headers = [
            "id", "title", "category", "x_coordinate", "y_coordinate", 
            "confidence", "impact_level", "time_horizon", "description"
        ]
        
        # Create CSV rows
        rows = [",".join(headers)]
        
        for point in radar_data:
            row = [
                str(point.get('id', '')),
                f'"{point.get("title", "")}"',  # Quote titles to handle commas
                str(point.get('category', '')),
                str(point.get('x', 0)),
                str(point.get('y', 0)),
                str(point.get('confidence', 0)),
                str(point.get('impact_level', '')),
                str(point.get('time_horizon_label', '')),
                f'"{point.get("description", "")}"'  # Quote descriptions
            ]
            rows.append(",".join(row))
        
        return "\n".join(rows)
    
    async def _create_svg_config(self, visualization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create SVG-specific configuration for static rendering"""
        
        config = visualization_data.get('visualization_config', {})
        radar_data = visualization_data.get('radar_data', [])
        
        svg_config = {
            "svg_dimensions": {
                "width": config.get('dimensions', {}).get('width', 800),
                "height": config.get('dimensions', {}).get('height', 600),
                "viewBox": f"0 0 {config.get('dimensions', {}).get('width', 800)} {config.get('dimensions', {}).get('height', 600)}"
            },
            "grid_lines": self._calculate_grid_lines(config),
            "axis_labels": config.get('axes', {}),
            "data_points": [
                {
                    "cx": self._scale_to_svg_x(point.get('x', 3), config),
                    "cy": self._scale_to_svg_y(point.get('y', 2), config),
                    "r": point.get('size', 10),
                    "fill": point.get('color', '#888888'),
                    "title": point.get('title', 'Unknown')
                }
                for point in radar_data
            ],
            "quadrant_backgrounds": self._create_quadrant_svg_elements(config)
        }
        
        return svg_config
    
    def _calculate_grid_lines(self, config: Dict[str, Any]) -> Dict[str, List]:
        """Calculate grid line positions for SVG"""
        dimensions = config.get('dimensions', {})
        width = dimensions.get('width', 800)
        height = dimensions.get('height', 600)
        margin = dimensions.get('margin', {"top": 50, "right": 50, "bottom": 50, "left": 50})
        
        chart_width = width - margin['left'] - margin['right']
        chart_height = height - margin['top'] - margin['bottom']
        
        # Vertical grid lines (time horizon)
        v_lines = []
        for i in range(1, 5):  # 1-4 scale
            x = margin['left'] + (i - 1) * chart_width / 3
            v_lines.append({"x1": x, "y1": margin['top'], "x2": x, "y2": height - margin['bottom']})
        
        # Horizontal grid lines (impact)
        h_lines = []
        for i in range(1, 5):  # 1-4 scale
            y = height - margin['bottom'] - (i - 1) * chart_height / 3
            h_lines.append({"x1": margin['left'], "y1": y, "x2": width - margin['right'], "y2": y})
        
        return {"vertical": v_lines, "horizontal": h_lines}
    
    def _scale_to_svg_x(self, x_value: float, config: Dict[str, Any]) -> float:
        """Scale x coordinate to SVG coordinate system"""
        dimensions = config.get('dimensions', {})
        width = dimensions.get('width', 800)
        margin = dimensions.get('margin', {"left": 50, "right": 50})
        
        chart_width = width - margin['left'] - margin['right']
        return margin['left'] + ((x_value - 1) / 3) * chart_width
    
    def _scale_to_svg_y(self, y_value: float, config: Dict[str, Any]) -> float:
        """Scale y coordinate to SVG coordinate system (inverted)"""
        dimensions = config.get('dimensions', {})
        height = dimensions.get('height', 600)
        margin = dimensions.get('margin', {"top": 50, "bottom": 50})
        
        chart_height = height - margin['top'] - margin['bottom']
        return height - margin['bottom'] - ((y_value - 1) / 3) * chart_height
    
    def _create_quadrant_svg_elements(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create SVG elements for quadrant backgrounds"""
        quadrants = config.get('quadrants', [])
        svg_quadrants = []
        
        for quadrant in quadrants:
            position = quadrant.get('position', {})
            
            x = self._scale_to_svg_x(position.get('x_min', 1), config)
            y = self._scale_to_svg_y(position.get('y_max', 4), config)  # SVG y is inverted
            width = self._scale_to_svg_x(position.get('x_max', 2), config) - x
            height = self._scale_to_svg_y(position.get('y_min', 1), config) - y
            
            svg_quadrants.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "fill": quadrant.get('color', 'rgba(0,0,0,0.05)'),
                "title": quadrant.get('name', 'Quadrant')
            })
        
        return svg_quadrants
    