"""
Trend Radar Orchestrator - Main coordination system for MCP agents.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from ..agents.data_collector import DataCollectorAgent
from ..agents.analysis_agent import AnalysisAgent
from ..agents.visualization_agent import VisualizationAgent
from ..agents.reporting_agent import ReportingAgent
from ..models.mcp_message import MCPMessage, MCPMessageType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TrendRadarOrchestrator:
    """Main orchestrator that coordinates all MCP agents"""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize orchestrator with agent coordination capabilities
        
        Args:
            llm_config: Configuration for LLM endpoints and models
        """
        self.session_id = str(uuid.uuid4())
        self.llm_config = llm_config or {
            "base_url": "http://localhost:11434",
            "model": "gpt-oss:20b"
        }
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Communication and coordination
        self.message_queue = asyncio.Queue()
        self.active_sessions = {}
        self.execution_history = []
        
        # Performance tracking
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_execution_time": 0.0,
            "agent_performance": {}
        }
        
        logger.info(f"TrendRadarOrchestrator initialized with session {self.session_id}")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all MCP agents with configuration"""
        agents = {
            "data_collector": DataCollectorAgent(**self.llm_config),
            "analysis_agent": AnalysisAgent(**self.llm_config),
            "visualization_agent": VisualizationAgent(**self.llm_config),
            "reporting_agent": ReportingAgent(**self.llm_config)
        }
        
        # Verify agent health
        logger.info(f"Initialized {len(agents)} agents")
        return agents
    
    async def receive_message(self, message: MCPMessage) -> None:
        """
        Receive and queue messages from agents
        
        Args:
            message: MCP message from an agent
        """
        await self.message_queue.put(message)
        logger.debug(f"Received {message.message_type.value} from {message.agent_id}")
    
    async def orchestrate_trend_analysis(
        self, 
        query: str = "emerging technology trends",
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration workflow for comprehensive trend analysis
        
        Args:
            query: Analysis query/topic
            analysis_config: Optional configuration for analysis depth and focus
            
        Returns:
            Complete analysis results with visualization and reports
        """
        start_time = datetime.now()
        correlation_id = str(uuid.uuid4())
        
        # Default configuration
        config = {
            "depth": "standard",  # light, standard, deep
            "focus_areas": ["impact", "confidence", "timeline"],
            "report_type": "comprehensive",
            "target_audience": "leadership"
        }
        config.update(analysis_config or {})
        
        logger.info(f"Starting trend analysis orchestration: '{query}'")
        logger.info(f"Configuration: {config}")
        
        try:
            # Check agent health before starting
            await self._verify_agent_health()
            
            # Execute analysis pipeline
            results = await self._execute_analysis_pipeline(query, config, correlation_id)
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            results["execution_metadata"] = {
                "session_id": self.session_id,
                "correlation_id": correlation_id,
                "execution_time_seconds": execution_time,
                "query": query,
                "config": config,
                "completed_at": datetime.now().isoformat()
            }
            
            # Update performance metrics
            self._update_performance_metrics(execution_time, True)
            
            logger.info(f"Analysis completed successfully in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Analysis orchestration failed: {str(e)}")
            self._update_performance_metrics(0, False)
            
            return {
                "error": str(e),
                "processing_complete": False,
                "session_id": self.session_id,
                "correlation_id": correlation_id,
                "failed_at": datetime.now().isoformat()
            }
    
    async def _verify_agent_health(self) -> None:
        """Verify all agents are healthy and responsive"""
        health_checks = []
        
        for agent_id, agent in self.agents.items():
            try:
                health_status = await agent.health_check()
                health_checks.append((agent_id, health_status))
                
                if not health_status:
                    logger.warning(f"Agent {agent_id} health check failed")
            except Exception as e:
                logger.error(f"Health check error for {agent_id}: {e}")
                health_checks.append((agent_id, False))
        
        failed_agents = [agent_id for agent_id, status in health_checks if not status]
        
        if failed_agents:
            raise RuntimeError(f"Agent health check failed for: {', '.join(failed_agents)}")
        
        logger.info("All agents passed health checks")
    
    async def _execute_analysis_pipeline(
        self, 
        query: str, 
        config: Dict[str, Any], 
        correlation_id: str
    ) -> Dict[str, Any]:
        """Execute the complete analysis pipeline with all agents"""
        
        pipeline_results = {}
        
        # Stage 1: Data Collection
        logger.info("Stage 1: Data Collection")
        collection_task = {
            "query": query,
            "depth": config.get("depth", "standard"),
            "sources": config.get("sources"),
            "task_id": f"{correlation_id}_collection"
        }
        
        collection_result = await self.agents["data_collector"].process_task(collection_task)
        pipeline_results["data_collection"] = collection_result
        
        # Stage 2: Trend Analysis
        logger.info("Stage 2: Trend Analysis")
        analysis_task = {
            "raw_trends": collection_result.get("raw_trends", []),
            "depth": config.get("depth", "standard"),
            "focus_areas": config.get("focus_areas", ["impact", "confidence", "timeline"]),
            "task_id": f"{correlation_id}_analysis"
        }
        
        analysis_result = await self.agents["analysis_agent"].process_task(analysis_task)
        pipeline_results["trend_analysis"] = analysis_result
        
        # Stage 3: Visualization Generation
        logger.info("Stage 3: Visualization Generation")
        visualization_task = {
            "analyzed_trends": analysis_result.get("analyzed_trends", []),
            "type": config.get("viz_type", "radar"),
            "config": config.get("viz_config", {}),
            "task_id": f"{correlation_id}_visualization"
        }
        
        visualization_result = await self.agents["visualization_agent"].process_task(visualization_task)
        pipeline_results["visualization"] = visualization_result
        
        # Stage 4: Report Generation
        logger.info("Stage 4: Report Generation")
        reporting_task = {
            "radar_data": visualization_result.get("radar_data", []),
            "statistics": visualization_result.get("statistics", {}),
            "supporting_charts": visualization_result.get("supporting_charts", {}),
            "type": config.get("report_type", "comprehensive"),
            "audience": config.get("target_audience", "leadership"),
            "task_id": f"{correlation_id}_reporting"
        }
        
        reporting_result = await self.agents["reporting_agent"].process_task(reporting_task)
        pipeline_results["report"] = reporting_result
        
        # Compile final integrated results
        integrated_results = self._integrate_pipeline_results(pipeline_results, config)
        
        return integrated_results
    
    def _integrate_pipeline_results(
        self, 
        pipeline_results: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate results from all pipeline stages"""
        
        # Extract key components
        data_collection = pipeline_results.get("data_collection", {})
        trend_analysis = pipeline_results.get("trend_analysis", {})
        visualization = pipeline_results.get("visualization", {})
        report = pipeline_results.get("report", {})
        
        # Create integrated result structure
        integrated_results = {
            "processing_complete": True,
            "trend_radar": {
                "data": visualization.get("radar_data", []),
                "configuration": visualization.get("visualization_config", {}),
                "supporting_charts": visualization.get("supporting_charts", {}),
                "statistics": visualization.get("statistics", {})
            },
            "analysis": {
                "analyzed_trends": trend_analysis.get("analyzed_trends", []),
                "cross_analysis": trend_analysis.get("cross_analysis", {}),
                "analysis_summary": trend_analysis.get("analysis_summary", {}),
                "quality_metrics": trend_analysis.get("quality_metrics", {})
            },
            "report": report,
            "data_collection_metadata": {
                "trends_collected": len(data_collection.get("raw_trends", [])),
                "sources_queried": data_collection.get("collection_metadata", {}).get("sources_queried", []),
                "collection_quality": data_collection.get("quality_metrics", {})
            },
            "pipeline_summary": {
                "stages_completed": len(pipeline_results),
                "total_trends_processed": len(trend_analysis.get("analyzed_trends", [])),
                "visualization_points": len(visualization.get("radar_data", [])),
                "insights_generated": len(report.get("key_insights", [])),
                "recommendations_provided": len(report.get("strategic_recommendations", []))
            }
        }
        
        return integrated_results
    
    def _update_performance_metrics(self, execution_time: float, success: bool) -> None:
        """Update orchestrator performance metrics"""
        self.performance_metrics["total_analyses"] += 1
        
        if success:
            self.performance_metrics["successful_analyses"] += 1
            
            # Update average execution time
            current_avg = self.performance_metrics["average_execution_time"]
            total_successful = self.performance_metrics["successful_analyses"]
            
            new_avg = ((current_avg * (total_successful - 1)) + execution_time) / total_successful
            self.performance_metrics["average_execution_time"] = round(new_avg, 2)
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        agent_statuses = {}
        
        for agent_id, agent in self.agents.items():
            try:
                status = agent.get_status()
                health = await agent.health_check()
                
                agent_statuses[agent_id] = {
                    **status,
                    "health_status": "healthy" if health else "unhealthy"
                }
            except Exception as e:
                agent_statuses[agent_id] = {
                    "status": "error",
                    "error": str(e),
                    "health_status": "unhealthy"
                }
        
        return {
            "session_id": self.session_id,
            "agents": agent_statuses,
            "performance_metrics": self.performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    async def export_results(
        self, 
        results: Dict[str, Any], 
        export_format: str = "json",
        filename: Optional[str] = None
    ) -> str:
        """
        Export analysis results in various formats
        
        Args:
            results: Analysis results to export
            export_format: Format to export (json, csv, html)
            filename: Optional filename, will generate if not provided
            
        Returns:
            Filepath of exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not filename:
            session_short = self.session_id[:8]
            filename = f"trend_radar_results_{session_short}_{timestamp}"
        
        if export_format.lower() == "json":
            filepath = f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        elif export_format.lower() == "csv":
            filepath = f"{filename}.csv"
            await self._export_to_csv(results, filepath)
        
        elif export_format.lower() == "html":
            filepath = f"{filename}.html"
            await self._export_to_html(results, filepath)
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Results exported to {filepath}")
        return filepath
    
    async def _export_to_csv(self, results: Dict[str, Any], filepath: str) -> None:
        """Export radar data to CSV format"""
        radar_data = results.get("trend_radar", {}).get("data", [])
        
        if not radar_data:
            logger.warning("No radar data to export to CSV")
            return
        
        # Use visualization agent to convert to CSV
        csv_data = self.agents["visualization_agent"]._convert_to_csv_format(radar_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(csv_data)
    
    async def _export_to_html(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to HTML report format"""
        
        # Extract key data
        report = results.get("report", {})
        radar_data = results.get("trend_radar", {}).get("data", [])
        statistics = results.get("trend_radar", {}).get("statistics", {})
        
        # Generate HTML content
        html_content = self._generate_html_report(report, radar_data, statistics)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_report(
        self, 
        report: Dict[str, Any], 
        radar_data: List[Dict[str, Any]], 
        statistics: Dict[str, Any]
    ) -> str:
        """Generate HTML report content"""
        
        # Basic HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trend Radar Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .trend {{ background: #f9f9f9; margin: 10px 0; padding: 10px; border-radius: 3px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e7f3ff; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trend Radar Analysis Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Session: {self.session_id[:8]}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{report.get('executive_summary', 'No executive summary available')}</p>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
                <div class="metric">Total Trends: {len(radar_data)}</div>
                <div class="metric">Avg Confidence: {statistics.get('overview', {}).get('average_confidence', 'N/A')}</div>
                <div class="metric">Avg Impact: {statistics.get('overview', {}).get('average_impact', 'N/A')}</div>
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                {self._format_insights_html(report.get('key_insights', []))}
            </div>
            
            <div class="section">
                <h2>Strategic Recommendations</h2>
                {self._format_recommendations_html(report.get('strategic_recommendations', []))}
            </div>
            
            <div class="section">
                <h2>Trend Details</h2>
                {self._format_trends_html(radar_data)}
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _format_insights_html(self, insights: List[Dict[str, Any]]) -> str:
        """Format insights for HTML display"""
        if not insights:
            return "<p>No insights available</p>"
        
        html_parts = []
        for insight in insights[:5]:  # Top 5 insights
            html_parts.append(f"""
            <div class="trend">
                <h4>{insight.get('title', 'Unknown Insight')}</h4>
                <p>{insight.get('description', 'No description available')}</p>
                <small>Type: {insight.get('type', 'unknown')} | Importance: {insight.get('importance', 'N/A')}</small>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _format_recommendations_html(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format recommendations for HTML display"""
        if not recommendations:
            return "<p>No recommendations available</p>"
        
        html_parts = []
        for rec in recommendations[:5]:  # Top 5 recommendations
            html_parts.append(f"""
            <div class="trend">
                <h4>{rec.get('title', 'Unknown Recommendation')}</h4>
                <p>{rec.get('description', 'No description available')}</p>
                <small>Priority: {rec.get('priority', 'medium')} | Timeframe: {rec.get('timeframe', 'N/A')} | Effort: {rec.get('effort', 'N/A')}</small>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _format_trends_html(self, radar_data: List[Dict[str, Any]]) -> str:
        """Format trend data as HTML table"""
        if not radar_data:
            return "<p>No trend data available</p>"
        
        table_rows = []
        for trend in radar_data[:10]:  # Top 10 trends
            table_rows.append(f"""
            <tr>
                <td>{trend.get('title', 'Unknown')}</td>
                <td>{trend.get('category', 'N/A')}</td>
                <td>{trend.get('y', 'N/A')}</td>
                <td>{trend.get('confidence', 'N/A')}</td>
                <td>{trend.get('time_horizon_label', 'N/A')}</td>
            </tr>
            """)
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Trend</th>
                    <th>Category</th>
                    <th>Impact</th>
                    <th>Confidence</th>
                    <th>Time Horizon</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
        """
    
    async def cleanup_session(self) -> None:
        """Cleanup session resources"""
        logger.info(f"Cleaning up session {self.session_id}")
        
        # Clear message queue
        while not self.message_queue.empty():
            try:
                await self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Reset agent states if needed
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
        
        logger.info("Session cleanup completed")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the orchestrator"""
        return {
            "session_id": self.session_id,
            "performance_metrics": self.performance_metrics.copy(),
            "success_rate": (
                self.performance_metrics["successful_analyses"] / 
                max(1, self.performance_metrics["total_analyses"])
            ),
            "agent_count": len(self.agents),
            "session_created": datetime.now().isoformat()  # This would be tracked in real implementation
        }
    