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
            "data_collector": DataCollectorAgent(
                llm_base_url=self.llm_config["base_url"],
                model_name=self.llm_config["model"]
            ),
            "analysis_agent": AnalysisAgent(
                llm_base_url=self.llm_config["base_url"], 
                model_name=self.llm_config["model"]
            ),
            "visualization_agent": VisualizationAgent(
                llm_base_url=self.llm_config["base_url"],
                model_name=self.llm_config["model"]
            ),
            "reporting_agent": ReportingAgent(
                llm_base_url=self.llm_config["base_url"],
                model_name=self.llm_config["model"]
            )
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
            logger.info("Checking agent health...")
            await self._verify_agent_health()
            
            # Execute analysis pipeline
            logger.info("Executing analysis pipeline...")
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
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            self._update_performance_metrics(0, False)
            
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_complete": False,
                "session_id": self.session_id,
                "correlation_id": correlation_id,
                "failed_at": datetime.now().isoformat(),
                "traceback": traceback.format_exc()
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
        
        try:
            # Stage 1: Data Collection
            logger.info("Stage 1: Data Collection")
            collection_task = {
                "query": query,
                "depth": config.get("depth", "standard"),
                "sources": config.get("sources"),
                "task_id": f"{correlation_id}_collection"
            }
            
            logger.debug(f"Data collection task: {collection_task}")
            collection_result = await self.agents["data_collector"].process_task(collection_task)
            logger.info(f"Data collection completed: {len(collection_result.get('raw_trends', []))} trends found")
            pipeline_results["data_collection"] = collection_result
            
        except Exception as e:
            logger.error(f"Data collection stage failed: {e}")
            raise RuntimeError(f"Data collection failed: {e}") from e
        
        try:
            # Stage 2: Trend Analysis
            logger.info("Stage 2: Trend Analysis")
            analysis_task = {
                "raw_trends": collection_result.get("raw_trends", []),
                "depth": config.get("depth", "standard"),
                "focus_areas": config.get("focus_areas", ["impact", "confidence", "timeline"]),
                "task_id": f"{correlation_id}_analysis"
            }
            
            logger.debug(f"Analysis task: {len(analysis_task['raw_trends'])} trends to analyze")
            analysis_result = await self.agents["analysis_agent"].process_task(analysis_task)
            logger.info(f"Trend analysis completed: {len(analysis_result.get('analyzed_trends', []))} trends analyzed")
            pipeline_results["trend_analysis"] = analysis_result
            
        except Exception as e:
            logger.error(f"Trend analysis stage failed: {e}")
            raise RuntimeError(f"Trend analysis failed: {e}") from e
        
        try:
            # Stage 3: Visualization Generation
            logger.info("Stage 3: Visualization Generation")
            visualization_task = {
                "analyzed_trends": analysis_result.get("analyzed_trends", []),
                "type": config.get("viz_type", "radar"),
                "config": config.get("viz_config", {}),
                "task_id": f"{correlation_id}_visualization"
            }
            
            logger.debug(f"Visualization task: {len(visualization_task['analyzed_trends'])} trends to visualize")
            visualization_result = await self.agents["visualization_agent"].process_task(visualization_task)
            logger.info(f"Visualization completed: {len(visualization_result.get('radar_data', []))} radar points generated")
            pipeline_results["visualization"] = visualization_result
            
        except Exception as e:
            logger.error(f"Visualization stage failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
        
        try:
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
            
            logger.debug(f"Reporting task: {len(reporting_task['radar_data'])} data points to report on")
            reporting_result = await self.agents["reporting_agent"].process_task(reporting_task)
            logger.info(f"Report generation completed: {len(reporting_result.get('key_insights', []))} insights generated")
            pipeline_results["report"] = reporting_result
            
        except Exception as e:
            logger.error(f"Report generation stage failed: {e}")
            raise RuntimeError(f"Report generation failed: {e}") from e
        
        # Compile final integrated results
        try:
            logger.info("Integrating pipeline results...")
            integrated_results = self._integrate_pipeline_results(pipeline_results, config)
            logger.info("Pipeline integration completed successfully")
            return integrated_results
            
        except Exception as e:
            logger.error(f"Pipeline integration failed: {e}")
            raise RuntimeError(f"Pipeline integration failed: {e}") from e
    
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
        """Export results to HTML report format with embedded plots"""
        
        # Extract key data
        report = results.get("report", {})
        radar_data = results.get("trend_radar", {}).get("data", [])
        statistics = results.get("trend_radar", {}).get("statistics", {})
        plot_files = results.get("trend_radar", {}).get("plot_files", {})
        supporting_plot_files = results.get("trend_radar", {}).get("supporting_plot_files", {})
        
        # Generate HTML content with embedded plots
        html_content = self._generate_html_report_with_plots(
            report, radar_data, statistics, plot_files, supporting_plot_files
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_report_with_plots(
        self, 
        report: Dict[str, Any], 
        radar_data: List[Dict[str, Any]], 
        statistics: Dict[str, Any],
        plot_files: Dict[str, str],
        supporting_plot_files: Dict[str, str]
    ) -> str:
        """Generate HTML report content with embedded plot images"""
        
        # Convert plot files to base64 for embedding
        embedded_plots = self._embed_plot_images(plot_files, supporting_plot_files)
        
        # Get analysis metadata
        pipeline_summary = report.get("pipeline_summary", {})
        insights = report.get("key_insights", [])
        recommendations = report.get("strategic_recommendations", [])
        
        # Enhanced HTML template with embedded plots
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trend Radar Analysis Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; padding: 20px; line-height: 1.6; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1200px; margin: 0 auto; 
                    background: white; border-radius: 15px; 
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{ 
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white; padding: 40px; text-align: center;
                }}
                .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em; }}
                .content {{ padding: 30px; }}
                .section {{ 
                    margin: 30px 0; padding: 25px; 
                    border-radius: 10px; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                }}
                .executive-summary {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }}
                .metrics-section {{ background: #f8f9ff; }}
                .insights-section {{ background: #fff5f5; }}
                .recommendations-section {{ background: #f0fff4; }}
                .plots-section {{ background: #fafafa; }}
                .trend-details-section {{ background: #f9f9f9; }}
                
                h2 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                .executive-summary h2 {{ color: white; border-bottom: 3px solid rgba(255,255,255,0.3); }}
                
                .metric {{ 
                    display: inline-block; margin: 15px; padding: 20px; 
                    background: white; border-radius: 10px; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    min-width: 120px; text-align: center;
                }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
                .metric-label {{ color: #7f8c8d; font-size: 0.9em; }}
                
                .trend {{ 
                    background: white; margin: 15px 0; padding: 20px; 
                    border-radius: 10px; border-left: 4px solid #3498db;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}
                .trend h4 {{ margin: 0 0 10px 0; color: #2c3e50; }}
                .trend p {{ margin: 10px 0; color: #34495e; }}
                .trend small {{ color: #7f8c8d; }}
                
                .insight {{ 
                    background: white; margin: 15px 0; padding: 20px; 
                    border-radius: 10px; border-left: 4px solid #e74c3c;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}
                
                .recommendation {{ 
                    background: white; margin: 15px 0; padding: 20px; 
                    border-radius: 10px; border-left: 4px solid #27ae60;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}
                
                .priority-high {{ border-left-color: #e74c3c !important; }}
                .priority-medium {{ border-left-color: #f39c12 !important; }}
                .priority-low {{ border-left-color: #3498db !important; }}
                
                .plot-container {{ 
                    text-align: center; margin: 25px 0; 
                    background: white; border-radius: 10px; 
                    padding: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .plot-container img {{ 
                    max-width: 100%; height: auto; 
                    border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .plot-title {{ 
                    font-size: 1.2em; font-weight: bold; 
                    margin-bottom: 15px; color: #2c3e50;
                }}
                
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
                
                .interactive-note {{
                    background: #e8f4fd; border: 1px solid #3498db;
                    border-radius: 10px; padding: 15px; margin: 15px 0;
                    text-align: center; color: #2980b9;
                }}
                
                .footer {{ 
                    background: #34495e; color: white; 
                    padding: 20px; text-align: center; 
                    font-size: 0.9em; opacity: 0.8;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Trend Radar Analysis Report</h1>
                    <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
                    <p>Session: {self.session_id[:8]}</p>
                </div>
                
                <div class="content">
                    <div class="section executive-summary">
                        <h2>📋 Executive Summary</h2>
                        <p>{report.get('executive_summary', 'No executive summary available')}</p>
                    </div>
                    
                    <div class="section metrics-section">
                        <h2>📊 Key Metrics</h2>
                        <div class="metric">
                            <div class="metric-value">{len(radar_data)}</div>
                            <div class="metric-label">Total Trends</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{statistics.get('overview', {}).get('average_confidence', 0):.0%}</div>
                            <div class="metric-label">Avg Confidence</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{statistics.get('overview', {}).get('average_impact', 0):.1f}/4</div>
                            <div class="metric-label">Avg Impact</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{len(insights)}</div>
                            <div class="metric-label">Key Insights</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{len(recommendations)}</div>
                            <div class="metric-label">Recommendations</div>
                        </div>
                    </div>
                    
                    {self._generate_plots_section_html(embedded_plots)}
                    
                    <div class="section insights-section">
                        <h2>💡 Key Insights</h2>
                        {self._format_insights_html_enhanced(insights)}
                    </div>
                    
                    <div class="section recommendations-section">
                        <h2>🎯 Strategic Recommendations</h2>
                        {self._format_recommendations_html_enhanced(recommendations)}
                    </div>
                    
                    <div class="section trend-details-section">
                        <h2>📈 Trend Details</h2>
                        {self._format_trends_html_enhanced(radar_data[:10])}
                    </div>
                </div>
                
                <div class="footer">
                    <p>Generated by Trend Radar MCP Application | AI-Powered Multi-Agent Analysis</p>
                    <p>Plots are embedded for offline viewing | Interactive versions available separately</p>
                </div>
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
    
    def _embed_plot_images(self, plot_files: Dict[str, str], supporting_plot_files: Dict[str, str]) -> Dict[str, str]:
        """Convert plot image files to base64 for HTML embedding"""
        import base64
        
        embedded_plots = {}
        
        # Combine all plot files
        all_plots = {**plot_files, **supporting_plot_files}
        
        for plot_name, plot_path in all_plots.items():
            if plot_path and Path(plot_path).exists() and plot_path.endswith('.png'):
                try:
                    with open(plot_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                        embedded_plots[plot_name] = f"data:image/png;base64,{image_data}"
                    
                    logger.debug(f"Embedded plot: {plot_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to embed plot {plot_name}: {e}")
                    embedded_plots[plot_name] = None
        
        return embedded_plots
    
    def _generate_plots_section_html(self, embedded_plots: Dict[str, str]) -> str:
        """Generate the plots section with embedded images"""
        
        if not embedded_plots:
            return """
            <div class="section plots-section">
                <h2>📊 Visualizations</h2>
                <p>No plots were generated for this analysis.</p>
            </div>
            """
        
        plots_html = """
        <div class="section plots-section">
            <h2>📊 Visualizations</h2>
            <div class="interactive-note">
                📱 <strong>Interactive versions:</strong> Look for *_interactive.html files in the plots directory for fully interactive charts!
            </div>
        """
        
        # Main radar plot
        if 'main_radar' in embedded_plots and embedded_plots['main_radar']:
            plots_html += f"""
            <div class="plot-container">
                <div class="plot-title">🎯 Main Trend Radar Chart</div>
                <img src="{embedded_plots['main_radar']}" alt="Trend Radar Chart" />
                <p style="margin-top: 15px; color: #7f8c8d; font-style: italic;">
                    Interactive quadrant view showing trends positioned by impact level (Y-axis) and time horizon (X-axis). 
                    Point size represents confidence level.
                </p>
            </div>
            """
        
        # Supporting plots in a grid
        supporting_plots = {k: v for k, v in embedded_plots.items() if k != 'main_radar' and v}
        
        if supporting_plots:
            plots_html += """
            <div class="grid">
            """
            
            plot_titles = {
                'category_distribution': '🥧 Category Distribution',
                'confidence_vs_impact': '📊 Confidence vs Impact Analysis', 
                'timeline_distribution': '📈 Timeline Distribution',
                'dashboard': '📋 Summary Dashboard'
            }
            
            for plot_name, plot_data in supporting_plots.items():
                title = plot_titles.get(plot_name, plot_name.replace('_', ' ').title())
                plots_html += f"""
                <div class="plot-container">
                    <div class="plot-title">{title}</div>
                    <img src="{plot_data}" alt="{title}" />
                </div>
                """
            
            plots_html += "</div>"
        
        plots_html += "</div>"
        return plots_html
    
    def _format_insights_html_enhanced(self, insights: List[Dict[str, Any]]) -> str:
        """Format insights for enhanced HTML display"""
        if not insights:
            return "<p>No insights available</p>"
        
        html_parts = []
        for i, insight in enumerate(insights[:8], 1):  # Show top 8 insights
            importance = insight.get('importance', 0.5)
            importance_color = '#e74c3c' if importance >= 0.8 else '#f39c12' if importance >= 0.6 else '#3498db'
            
            html_parts.append(f"""
            <div class="insight" style="border-left-color: {importance_color};">
                <h4>💡 {insight.get('title', 'Unknown Insight')}</h4>
                <p>{insight.get('description', 'No description available')}</p>
                <small>
                    <strong>Type:</strong> {insight.get('type', 'unknown').title()} | 
                    <strong>Importance:</strong> {importance:.0%} | 
                    <strong>Category:</strong> {insight.get('category', 'general').title()}
                </small>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _format_recommendations_html_enhanced(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format recommendations for enhanced HTML display"""
        if not recommendations:
            return "<p>No recommendations available</p>"
        
        html_parts = []
        for i, rec in enumerate(recommendations[:8], 1):  # Show top 8 recommendations
            priority = rec.get('priority', 'medium').lower()
            priority_class = f"priority-{priority}"
            
            priority_icons = {'high': '🚨', 'medium': '⚠️', 'low': '💡'}
            icon = priority_icons.get(priority, '📝')
            
            html_parts.append(f"""
            <div class="recommendation {priority_class}">
                <h4>{icon} {rec.get('title', 'Unknown Recommendation')}</h4>
                <p>{rec.get('description', 'No description available')}</p>
                <small>
                    <strong>Priority:</strong> {priority.title()} | 
                    <strong>Timeframe:</strong> {rec.get('timeframe', 'N/A')} | 
                    <strong>Effort:</strong> {rec.get('effort', 'N/A')} |
                    <strong>Impact:</strong> {rec.get('impact', 'N/A')}
                </small>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _format_trends_html_enhanced(self, radar_data: List[Dict[str, Any]]) -> str:
        """Format trend data as enhanced HTML"""
        if not radar_data:
            return "<p>No trend data available</p>"
        
        html_parts = []
        
        for trend in radar_data:
            confidence = trend.get('confidence', 0.5)
            impact_level = trend.get('impact_level', 'medium')
            
            # Color coding based on impact
            impact_colors = {
                'critical': '#e74c3c', 'high': '#f39c12', 
                'medium': '#3498db', 'low': '#95a5a6'
            }
            impact_color = impact_colors.get(impact_level, '#3498db')
            
            # Build sources and URLs info
            sources_info = ""
            if trend.get('sources'):
                sources_info += f"<strong>Sources:</strong> {', '.join(trend['sources'][:3])} "
            if trend.get('urls'):
                urls_count = len(trend['urls'])
                sources_info += f"({urls_count} reference{'s' if urls_count != 1 else ''})"
            
            html_parts.append(f"""
            <div class="trend" style="border-left-color: {impact_color};">
                <h4>{trend.get('title', 'Unknown Trend')}</h4>
                <p>{trend.get('description', 'No description available')}</p>
                <small>
                    <strong>Category:</strong> {trend.get('category', 'N/A').title()} | 
                    <strong>Impact:</strong> {impact_level.title()} | 
                    <strong>Confidence:</strong> {confidence:.0%} | 
                    <strong>Timeline:</strong> {trend.get('time_horizon_label', 'N/A')}
                    <br>{sources_info}
                </small>
            </div>
            """)
        
        return "".join(html_parts)
    
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
    