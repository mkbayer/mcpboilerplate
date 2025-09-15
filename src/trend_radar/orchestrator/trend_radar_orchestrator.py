"""
Trend Radar Orchestrator - Main coordination system for MCP agents.
"""

import asyncio
import uuid
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
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
            "model": "gpt-oss:20b"
        }
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Communication and coordination
        self.message_queue = asyncio.Queue()
        self.active_sessions = {}
        self.execution_history = []
        
        # Performance tracking
        self._metrics = {"runs": 0, "success": 0, "failures": 0, "total_time": 0.0}
    

    def _initialize_agents(self) -> Dict[str, Any]:
        """Instantiate agent classes and return mapping. Must not return None."""
        try:
            base_url = self.llm_config.get("base_url", "http://localhost:11434")
            model = self.llm_config.get("model", "gpt-oss:20b")
            agents = {
                "data_collector": DataCollectorAgent(llm_base_url=base_url, model_name=model),
                "analysis_agent": AnalysisAgent(llm_base_url=base_url, model_name=model),
                "visualization_agent": VisualizationAgent(llm_base_url=base_url, model_name=model),
                "reporting_agent": ReportingAgent(llm_base_url=base_url, model_name=model),
            }
            logger.info(f"Initialized {len(agents)} agents")
            return agents
        except Exception as e:
            logger.exception("Failed to initialize agents")
            raise RuntimeError(f"Agent initialization failed: {e}") from e
    
    
    async def receive_message(self, message: MCPMessage) -> None:
        """Put incoming MCPMessage into internal queue for processing."""
        await self.message_queue.put(message)
        logger.debug("Message queued", extra={"message_id": getattr(message, "id", None)})
    
    
    async def orchestrate_trend_analysis(
        self, 
        query: str = "emerging technology trends",
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Public entrypoint for running the full analysis pipeline."""
        config = analysis_config or {}
        correlation_id = f"{self.session_id}-{int(datetime.utcnow().timestamp())}"
        start = datetime.utcnow()
        try:
            await self._verify_agent_health()
            result = await self._execute_analysis_pipeline(query, config, correlation_id)
            elapsed = (datetime.utcnow() - start).total_seconds()
            self._update_performance_metrics(elapsed, True)
            return result
        except Exception as e:
            elapsed = (datetime.utcnow() - start).total_seconds()
            self._update_performance_metrics(elapsed, False)
            logger.error(f"Analysis orchestration failed: {e}")
            raise


    async def _verify_agent_health(self) -> None:
        """Lightweight health check for agents. Raises if any agent is missing."""
        if not isinstance(self.agents, dict) or len(self.agents) == 0:
            raise RuntimeError("No agents initialized")
        # optional deeper checks if agents expose health/status methods
        for name, agent in self.agents.items():
            if agent is None:
                raise RuntimeError(f"Agent {name} is None")


    async def _execute_analysis_pipeline(
        self, 
        query: str, 
        config: Dict[str, Any], 
        correlation_id: str
    ) -> Dict[str, Any]:
        """Execute the complete analysis pipeline with robust error handling."""
        
        pipeline_results: Dict[str, Any] = {}
        
        def _ensure_agent(name: str):
            agent = self.agents.get(name)
            if agent is None:
                raise RuntimeError(f"Agent '{name}' not initialized or missing")
            if not hasattr(agent, "process_task"):
                raise RuntimeError(f"Agent '{name}' does not implement process_task")
            return agent

        def _ensure_result(result: Optional[Dict[str, Any]], stage: str) -> Dict[str, Any]:
            """Ensure we have a valid result dictionary"""
            if result is None:
                logger.error(f"{stage} returned None instead of dict")
                return {}  # Return empty dict instead of None
            if not isinstance(result, dict):
                logger.error(f"{stage} returned {type(result)} instead of dict")
                return {}
            return result

        try:
            # Stage 1: Data Collection
            logger.info("Stage 1: Data Collection")
            collection_task = {
                "query": query,
                "depth": config.get("depth", "standard"),
                "sources": config.get("sources", []),
                "task_id": f"{correlation_id}_collection"
            }
            collection_agent = _ensure_agent("data_collector")
            collection_result = _ensure_result(
                await collection_agent.process_task(collection_task),
                "Data collection"
            )
            pipeline_results["data_collection"] = collection_result

            # Stage 2: Trend Analysis
            logger.info("Stage 2: Trend Analysis")
            analysis_task = {
                "raw_trends": collection_result.get("raw_trends", []),
                "depth": config.get("depth", "standard"),
                "focus_areas": config.get("focus_areas", ["impact", "confidence", "timeline"]),
                "task_id": f"{correlation_id}_analysis"
            }
            analysis_agent = _ensure_agent("analysis_agent")
            analysis_result = _ensure_result(
                await analysis_agent.process_task(analysis_task),
                "Trend analysis"
            )
            pipeline_results["trend_analysis"] = analysis_result

            # Stage 3: Visualization Generation
            logger.info("Stage 3: Visualization Generation")
            visualization_task = {
                "analyzed_trends": analysis_result.get("analyzed_trends", []),
                "type": config.get("viz_type", "radar"),
                "config": config.get("viz_config", {}),
                "task_id": f"{correlation_id}_visualization"
            }
            viz_agent = _ensure_agent("visualization_agent")
            visualization_result = _ensure_result(
                await viz_agent.process_task(visualization_task),
                "Visualization generation"
            )
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
            report_agent = _ensure_agent("reporting_agent")
            reporting_result = _ensure_result(
                await report_agent.process_task(reporting_task),
                "Report generation"
            )
            pipeline_results["report"] = reporting_result

            return self._integrate_pipeline_results(pipeline_results, config)
            
        except Exception as e:
            logger.exception(f"Pipeline execution failed at stage {len(pipeline_results) + 1}")
            logger.error(f"Pipeline state: {json.dumps(pipeline_results, indent=2)}")
            raise RuntimeError(f"Analysis pipeline failed: {str(e)}") from e


    def _integrate_pipeline_results(
        self, 
        pipeline_results: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge stage outputs into a coherent return structure."""
        return {
            "metadata": {
                "run_id": str(uuid.uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "config": config
            },
            "data_collection": pipeline_results.get("data_collection", {}),
            "trend_analysis": pipeline_results.get("trend_analysis", {}),
            "visualization": pipeline_results.get("visualization", {}),
            "report": pipeline_results.get("report", {}),
        }
    
    
    def _update_performance_metrics(self, execution_time: float, success: bool) -> None:
        self._metrics["runs"] += 1
        self._metrics["total_time"] += execution_time
        if success:
            self._metrics["success"] += 1
        else:
            self._metrics["failures"] += 1
    
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Return basic status info for each agent (non-blocking)."""
        status = {}
        for name, agent in self.agents.items():
            try:
                # prefer sync attribute or method 'status' / 'get_status'
                if hasattr(agent, "get_status"):
                    maybe = agent.get_status()
                    if asyncio.iscoroutine(maybe):
                        maybe = await maybe
                    status[name] = maybe
                elif hasattr(agent, "status"):
                    status[name] = getattr(agent, "status")
                else:
                    status[name] = {"status": "unknown"}
            except Exception as e:
                status[name] = {"status": "error", "error": str(e)}
        return status
    
    
    async def export_results(
        self, 
        results: Dict[str, Any], 
        export_format: str = "json",
        filename: Optional[str] = None
    ) -> str:
        """Export integrated results to disk. Returns path to written file."""
        filename = filename or f"trend_report_{int(datetime.utcnow().timestamp())}"
        if export_format == "json":
            filepath = f"{filename}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            return filepath
        elif export_format == "csv":
            filepath = f"{filename}.csv"
            await self._export_to_csv(results, filepath)
            return filepath
        elif export_format == "html":
            filepath = f"{filename}.html"
            await self._export_to_html(results, filepath)
            return filepath
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    
    async def _export_to_csv(self, results: Dict[str, Any], filepath: str) -> None:
        """Simple CSV export for top-level items (best-effort)."""
        import csv
        # flatten some data for CSV output (best-effort)
        rows = []
        trends = results.get("visualization", {}).get("radar_data", []) or []
        if trends:
            keys = set().union(*(t.keys() for t in trends))
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(keys))
                writer.writeheader()
                writer.writerows(trends)
        else:
            # fallback: write metadata
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "value"])
                for k, v in results.get("metadata", {}).items():
                    writer.writerow([k, json.dumps(v)])
    
    
    async def _export_to_html(self, results: Dict[str, Any], filepath: str) -> None:
        html = self._generate_html_report(results.get("report", {}), results.get("visualization", {}).get("radar_data", []), results.get("visualization", {}).get("statistics", {}))
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
    
    
    def _generate_html_report(
        self, 
        report: Dict[str, Any], 
        radar_data: List[Dict[str, Any]], 
        statistics: Dict[str, Any]
    ) -> str:
        """Minimal HTML generation (safe, simple)."""
        return f"""<html><head><meta charset="utf-8"><title>Trend Report</title></head>
        <body>
        <h1>{report.get('metadata', {}).get('title', 'Trend Radar Report')}</h1>
        <h2>Executive Summary</h2><pre>{report.get('executive_summary', '')}</pre>
        <h2>Key Insights</h2><pre>{json.dumps(report.get('key_insights', []), indent=2)}</pre>
        <h2>Recommendations</h2><pre>{json.dumps(report.get('strategic_recommendations', []), indent=2)}</pre>
        <h2>Trends</h2><pre>{json.dumps(radar_data[:50], indent=2)}</pre>
        </body></html>"""
    
    
    def _format_insights_html(self, insights: List[Dict[str, Any]]) -> str:
        return "\n".join(f"<li>{json.dumps(i)}</li>" for i in insights)
    
    
    def _format_recommendations_html(self, recommendations: List[Dict[str, Any]]) -> str:
        return "\n".join(f"<li>{json.dumps(r)}</li>" for r in recommendations)
    
    
    def _format_trends_html(self, radar_data: List[Dict[str, Any]]) -> str:
        return "\n".join(f"<li>{json.dumps(t)}</li>" for t in radar_data)
    
    
    async def cleanup_session(self) -> None:
        """Clear active sessions and message queue."""
        self.active_sessions.clear()
        while not self.message_queue.empty():
            try:
                _ = self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    
    def get_performance_summary(self) -> Dict[str, Any]:
        runs = self._metrics["runs"]
        avg_time = (self._metrics["total_time"] / runs) if runs else 0.0
        return {
            "runs": runs,
            "success": self._metrics["success"],
            "failures": self._metrics["failures"],
            "average_time_sec": avg_time
        }
