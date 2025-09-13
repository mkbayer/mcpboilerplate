# src/trend_radar/__init__.py
"""
🎯 Trend Radar MCP Application

A sophisticated Model Context Protocol demonstration showcasing AI-powered 
trend analysis through coordinated multi-agent orchestration.
"""

__version__ = "1.0.0"
__author__ = "Trend Radar Team"
__description__ = "AI-powered trend analysis with MCP agent orchestration"

from .orchestrator.trend_radar_orchestrator import TrendRadarOrchestrator
from .models.trend import Trend, TrendCategory, TrendImpact, TrendTimeHorizon
from .models.mcp_message import MCPMessage, MCPMessageType

__all__ = [
    "TrendRadarOrchestrator",
    "Trend", 
    "TrendCategory",
    "TrendImpact", 
    "TrendTimeHorizon",
    "MCPMessage",
    "MCPMessageType"
]

if __name__ == "__main__":
    print("TrendRadarOrchestrator:", TrendRadarOrchestrator)
    print("Trend:", Trend)
    print("MCPMessage:", MCPMessage)

