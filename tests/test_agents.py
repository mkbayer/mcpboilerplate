"""
Test suite for trend radar MCP agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from trend_radar.agents.data_collector import DataCollectorAgent
from trend_radar.agents.analysis_agent import AnalysisAgent
from trend_radar.agents.visualization_agent import VisualizationAgent
from trend_radar.agents.reporting_agent import ReportingAgent
from trend_radar.orchestrator.trend_radar_orchestrator import TrendRadarOrchestrator
from trend_radar.models.trend import Trend, TrendCategory, TrendImpact, TrendTimeHorizon
from trend_radar.models.mcp_message import MCPMessage, MCPMessageType


class TestDataCollectorAgent:
    """Test cases for DataCollectorAgent"""
    
    @pytest.fixture
    def data_collector(self):
        return DataCollectorAgent()
    
    def test_agent_initialization(self, data_collector):
        assert data_collector.agent_id == "data_collector"
        assert len(data_collector._define_capabilities()) == 4
        assert "tech_blogs" in data_collector.data_sources
    
    @pytest.mark.asyncio
    async def test_process_task(self, data_collector):
        # Mock LLM response
        with patch.object(data_collector, 'query_llm') as mock_llm:
            mock_llm.return_value = """
            Trend: Artificial Intelligence in Healthcare
            Description: AI systems transforming medical diagnosis and treatment
            Category: technology
            Keywords: AI, healthcare, diagnosis, machine learning
            """
            
            task = {"query": "healthcare technology trends"}
            result = await data_collector.process_task(task)
            
            assert "raw_trends" in result
            assert "collection_metadata" in result
            assert result["collection_metadata"]["query"] == "healthcare technology trends"
    
    def test_parse_trend_data(self, data_collector):
        response = """
        Trend: Quantum Computing Applications
        Description: Practical quantum computing solutions for enterprise
        Category: technology
        
        Trend: Remote Work Evolution  
        Description: New models of distributed work and collaboration
        Category: business
        """
        
        trends = data_collector._parse_trend_data(response)
        
        assert len(trends) == 2
        assert trends[0]["title"] == "Quantum Computing Applications"
        assert trends[1]["category"] == "business"


class TestAnalysisAgent:
    """Test cases for AnalysisAgent"""
    
    @pytest.fixture
    def analysis_agent(self):
        return AnalysisAgent()
    
    def test_agent_initialization(self, analysis_agent):
        assert analysis_agent.agent_id == "analysis_agent"
        assert len(analysis_agent._define_capabilities()) == 5
        assert "trend_impact_analysis" in [cap.name for cap in analysis_agent._define_capabilities()]
    
    @pytest.mark.asyncio
    async def test_process_task(self, analysis_agent):
        raw_trends = [
            {
                "title": "AI-Powered Automation",
                "description": "Artificial intelligence automating business processes",
                "category": "technology",
                "keywords": ["AI", "automation", "efficiency"]
            }
        ]
        
        with patch.object(analysis_agent, 'query_llm') as mock_llm:
            mock_llm.return_value = """
            IMPACT_LEVEL: HIGH
            MARKET_IMPACT: 8 - Significant market transformation expected
            CONFIDENCE_SCORE: 0.8
            TIME_HORIZON: SHORT_TERM
            """
            
            task = {"raw_trends": raw_trends}
            result = await analysis_agent.process_task(task)
            
            assert "analyzed_trends" in result
            assert "analysis_summary" in result
            assert len(result["analyzed_trends"]) == 1
    
    def test_assess_data_quality(self, analysis_agent):
        good_trend = {
            "title": "Comprehensive AI Trend",
            "description": "A detailed description of the artificial intelligence trend with substantial content",
            "keywords": ["AI", "machine learning", "automation"],
            "category": "technology"
        }
        
        poor_trend = {
            "title": "AI",
            "description": "Short",
            "keywords": [],
            "category": "unknown"
        }
        
        good_score = analysis_agent._assess_data_quality(good_trend)
        poor_score = analysis_agent._assess_data_quality(poor_trend)
        
        assert good_score > poor_score
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= poor_score <= 1.0


class TestVisualizationAgent:
    """Test cases for VisualizationAgent"""
    
    @pytest.fixture
    def viz_agent(self):
        return VisualizationAgent()
    
    def test_agent_initialization(self, viz_agent):
        assert viz_agent.agent_id == "visualization_agent"
        assert len(viz_agent._define_capabilities()) == 5
        assert "technology" in viz_agent.color_schemes
    
    @pytest.mark.asyncio
    async def test_create_radar_data(self, viz_agent):
        analyzed_trends = [
            {
                "id": "trend_1",
                "title": "Test Trend",
                "category": "technology",
                "impact_score": 3,
                "time_horizon": "short_term",
                "confidence_score": 0.8,
                "description": "A test trend for visualization"
            }
        ]
        
        radar_data = await viz_agent._create_radar_data(analyzed_trends)
        
        assert len(radar_data) == 1
        assert radar_data[0]["title"] == "Test Trend"
        assert radar_data[0]["x"] == 2  # short_term maps to 2
        assert radar_data[0]["y"] == 3  # impact_score
        assert radar_data[0]["size"] > 5  # confidence-based size
    
    def test_calculate_data_completeness(self, viz_agent):
        complete_trends = [
            {
                "title": "Complete Trend",
                "description": "Full description here",
                "category": "technology",
                "impact_score": 3,
                "confidence_score": 0.8,
                "time_horizon": "medium_term"
            }
        ]
        
        incomplete_trends = [
            {
                "title": "Incomplete",
                "description": "",
                "category": None,
                "impact_score": None
            }
        ]
        
        complete_score = viz_agent._calculate_data_completeness(complete_trends)
        incomplete_score = viz_agent._calculate_data_completeness(incomplete_trends)
        
        assert complete_score > incomplete_score
        assert complete_score == 1.0  # All fields complete
        assert incomplete_score < 0.5  # Many fields missing


class TestReportingAgent:
    """Test cases for ReportingAgent"""
    
    @pytest.fixture
    def reporting_agent(self):
        return ReportingAgent()
    
    def test_agent_initialization(self, reporting_agent):
        assert reporting_agent.agent_id == "reporting_agent"
        assert len(reporting_agent._define_capabilities()) == 5
        assert "executive_reporting" in [cap.name for cap in reporting_agent._define_capabilities()]
    
    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, reporting_agent):
        radar_data = [
            {
                "title": "AI Revolution",
                "category": "technology",
                "y": 4,  # high impact
                "confidence": 0.9,
                "description": "Artificial intelligence transforming industries"
            }
        ]
        
        statistics = {
            "overview": {
                "average_confidence": 0.85,
                "average_impact": 3.2
            }
        }
        
        with patch.object(reporting_agent, 'query_llm') as mock_llm:
            mock_llm.return_value = """
            Based on comprehensive trend analysis, the technology landscape shows 
            significant transformation potential. Key findings include high-impact 
            AI developments with strong confidence levels, suggesting immediate 
            strategic attention is required.
            """
            
            summary = await reporting_agent._generate_executive_summary(
                radar_data, statistics, "leadership"
            )
            
            assert isinstance(summary, str)
            assert len(summary) > 50
            assert "technology" in summary.lower() or "AI" in summary
    
    def test_rank_insights(self, reporting_agent):
        insights = [
            {
                "title": "High Impact Insight",
                "importance": 0.9,
                "description": "Very important finding"
            },
            {
                "title": "Medium Impact Insight", 
                "importance": 0.6,
                "description": "Moderately important finding"
            },
            {
                "title": "High Impact Duplicate",
                "importance": 0.9,
                "description": "High Impact similar content"  # Similar to first
            }
        ]
        
        ranked = reporting_agent._rank_insights(insights)
        
        # Should be ranked by importance and duplicates removed
        assert len(ranked) <= len(insights)
        assert ranked[0]["importance"] >= ranked[-1]["importance"]


class TestTrendRadarOrchestrator:
    """Test cases for TrendRadarOrchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        return TrendRadarOrchestrator()
    
    def test_orchestrator_initialization(self, orchestrator):
        assert orchestrator.session_id is not None
        assert len(orchestrator.agents) == 4
        assert "data_collector" in orchestrator.agents
        assert "analysis_agent" in orchestrator.agents
        assert "visualization_agent" in orchestrator.agents
        assert "reporting_agent" in orchestrator.agents
    
    @pytest.mark.asyncio
    async def test_receive_message(self, orchestrator):
        message = MCPMessage(
            agent_id="test_agent",
            message_type=MCPMessageType.STATUS_UPDATE,
            content={"status": "idle"},
            timestamp=datetime.now()
        )
        
        await orchestrator.receive_message(message)
        
        # Message should be in queue
        queued_message = await orchestrator.message_queue.get()
        assert queued_message.agent_id == "test_agent"
        assert queued_message.message_type == MCPMessageType.STATUS_UPDATE
    
    @pytest.mark.asyncio
    async def test_verify_agent_health(self, orchestrator):
        # Mock all agents to return healthy
        for agent in orchestrator.agents.values():
            agent.health_check = AsyncMock(return_value=True)
        
        # Should not raise exception for healthy agents
        await orchestrator._verify_agent_health()
        
        # Mock one agent as unhealthy
        orchestrator.agents["data_collector"].health_check = AsyncMock(return_value=False)
        
        # Should raise exception for unhealthy agent
        with pytest.raises(RuntimeError):
            await orchestrator._verify_agent_health()
    
    def test_update_performance_metrics(self, orchestrator):
        initial_total = orchestrator.performance_metrics["total_analyses"]
        initial_successful = orchestrator.performance_metrics["successful_analyses"]
        
        # Update with successful analysis
        orchestrator._update_performance_metrics(10.5, True)
        
        assert orchestrator.performance_metrics["total_analyses"] == initial_total + 1
        assert orchestrator.performance_metrics["successful_analyses"] == initial_successful + 1
        assert orchestrator.performance_metrics["average_execution_time"] > 0
        
        # Update with failed analysis
        orchestrator._update_performance_metrics(0, False)
        
        assert orchestrator.performance_metrics["total_analyses"] == initial_total + 2
        assert orchestrator.performance_metrics["successful_analyses"] == initial_successful + 1


class TestTrendModel:
    """Test cases for Trend data model"""
    
    def test_trend_creation(self):
        trend = Trend(
            id="test_1",
            title="Test Trend",
            description="A test trend for validation",
            category=TrendCategory.TECHNOLOGY,
            impact=TrendImpact.HIGH,
            time_horizon=TrendTimeHorizon.SHORT_TERM,
            confidence_score=0.85,
            sources=["test_source"],
            keywords=["test", "trend"],
            timestamp=datetime.now()
        )
        
        assert trend.id == "test_1"
        assert trend.title == "Test Trend"
        assert trend.category == TrendCategory.TECHNOLOGY
        assert trend.impact == TrendImpact.HIGH
        assert trend.confidence_score == 0.85
    
    def test_trend_to_dict(self):
        trend = Trend(
            id="test_1",
            title="Test Trend",
            description="A test trend",
            category=TrendCategory.BUSINESS,
            impact=TrendImpact.MEDIUM,
            time_horizon=TrendTimeHorizon.EMERGING,
            confidence_score=0.7,
            sources=["source1"],
            keywords=["keyword1"],
            timestamp=datetime.now()
        )
        
        trend_dict = trend.to_dict()
        
        assert trend_dict["id"] == "test_1"
        assert trend_dict["category"] == "business"  # Enum value
        assert trend_dict["impact"] == "medium"  # Enum value
        assert trend_dict["time_horizon"] == "emerging"  # Enum value
        assert "timestamp" in trend_dict


class TestMCPMessage:
    """Test cases for MCP message model"""
    
    def test_mcp_message_creation(self):
        message = MCPMessage(
            agent_id="test_agent",
            message_type=MCPMessageType.TASK_REQUEST,
            content={"task": "test_task", "data": "test_data"},
            timestamp=datetime.now(),
            correlation_id="test_correlation",
            priority=5
        )
        
        assert message.agent_id == "test_agent"
        assert message.message_type == MCPMessageType.TASK_REQUEST
        assert message.content["task"] == "test_task"
        assert message.priority == 5
    
    def test_mcp_message_to_dict(self):
        timestamp = datetime.now()
        message = MCPMessage(
            agent_id="test_agent",
            message_type=MCPMessageType.STATUS_UPDATE,
            content={"status": "active"},
            timestamp=timestamp
        )
        
        message_dict = message.to_dict()
        
        assert message_dict["agent_id"] == "test_agent"
        assert message_dict["message_type"] == "status_update"  # Enum value
        assert message_dict["content"]["status"] == "active"
        assert message_dict["timestamp"] == timestamp.isoformat()


# Integration test
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_basic_analysis_flow(self):
        """Test a simplified end-to-end analysis flow"""
        orchestrator = TrendRadarOrchestrator()
        
        # Mock LLM responses for all agents
        mock_responses = {
            "data_collector": """
            Trend: Test AI Trend
            Description: Artificial intelligence trend for testing
            Category: technology
            Keywords: AI, testing, automation
            """,
            "analysis_agent": """
            IMPACT_LEVEL: HIGH
            CONFIDENCE_SCORE: 0.8
            TIME_HORIZON: SHORT_TERM
            """,
            "visualization_agent": "Mock visualization response",
            "reporting_agent": """
            Executive Summary: The analysis reveals significant opportunities 
            in AI technology with high confidence levels and near-term impact potential.
            """
        }
        
        # Mock all agent LLM calls
        for agent_name, agent in orchestrator.agents.items():
            if agent_name in mock_responses:
                agent.query_llm = AsyncMock(return_value=mock_responses[agent_name])
        
        # Mock health checks
        for agent in orchestrator.agents.values():
            agent.health_check = AsyncMock(return_value=True)
        
        try:
            # Run analysis
            config = {
                "depth": "light",
                "report_type": "executive"
            }
            
            result = await orchestrator.orchestrate_trend_analysis(
                query="test AI trends",
                analysis_config=config
            )
            
            # Verify results structure
            assert result["processing_complete"] is True
            assert "trend_radar" in result
            assert "analysis" in result
            assert "report" in result
            assert "execution_metadata" in result
            
        finally:
            await orchestrator.cleanup_session()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    