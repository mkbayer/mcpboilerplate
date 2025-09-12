
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List
from mcp import MCPMessage, MCPAgent
from mcpboilerplate.src.trend_radar.agents.data_collector import DataCollectorAgent
from mcpboilerplate.src.trend_radar.agents.analysis_agent import AnalysisAgent
from mcpboilerplate.src.trend_radar.agents.visualization_agent import VisualizationAgent
from mcpboilerplate.src.trend_radar.agents.reporting_agent import ReportingAgent
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


