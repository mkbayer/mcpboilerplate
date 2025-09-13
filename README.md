# 🎯 Trend Radar MCP Application

A sophisticated **Model Context Protocol (MCP)** demonstration showcasing AI-powered trend analysis through coordinated multi-agent orchestration. This application uses four specialized agents to collect, analyze, visualize, and report on emerging trends using the **gpt-oss:20b** model.

## 🌟 Features

### 🤖 Four Specialized MCP Agents
- **DataCollectorAgent**: Gathers trend data from multiple sources
- **AnalysisAgent**: Analyzes trends for impact, confidence, and timeline
- **VisualizationAgent**: Creates trend radar charts and supporting visualizations  
- **ReportingAgent**: Generates comprehensive reports with insights and recommendations

### 🎯 Trend Radar Capabilities
- **Impact Assessment**: Low to Critical impact scoring
- **Confidence Analysis**: Data quality and reliability scoring
- **Timeline Prediction**: Emerging to Long-term horizon mapping
- **Strategic Quadrants**: Quick Wins, Strategic Bets, Tactical Moves, Background Noise
- **Risk/Opportunity Matrix**: Comprehensive risk and opportunity identification

### 📊 Output Formats
- **JSON**: Structured data for integration
- **CSV**: Tabular data for spreadsheet analysis
- **HTML**: Rich reports with visualizations
- **Interactive Dashboard**: Real-time trend radar visualization

## 🚀 Quick Start

### Prerequisites

1. **Install uv package manager**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install and setup Ollama**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull gpt-oss:20b
   ```

3. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd trend-radar-mcp
   ```

2. **Install dependencies with uv**:
   ```bash
   uv sync
   ```

3. **Install in development mode**:
   ```bash
   uv pip install -e .
   ```

### Basic Usage

1. **Run a quick analysis**:
   ```bash
   trend-radar analyze "artificial intelligence trends 2025"
   ```

2. **Interactive mode**:
   ```bash
   trend-radar interactive
   ```

3. **Health check**:
   ```bash
   trend-radar health-check
   ```

4. **Demo mode**:
   ```bash
   trend-radar demo
   ```

## 📖 Detailed Usage

### Command Line Interface

```bash
# Basic analysis with default settings
trend-radar analyze "sustainable technology innovations"

# Deep analysis with HTML output
trend-radar analyze "future of remote work" --depth deep --output-format html

# Executive report for leadership
trend-radar analyze "blockchain applications" --report-type executive --depth light

# Custom LLM configuration
trend-radar analyze "quantum computing trends" --llm-url http://localhost:11434 --model gpt-oss:20b
```

### Analysis Configuration Options

- **Depth Levels**:
  - `light`: 3-4 trends with brief analysis
  - `standard`: 5-7 trends with detailed analysis (default)
  - `deep`: 8-12 trends with comprehensive analysis

- **Report Types**:
  - `executive`: High-level summary for leadership
  - `detailed`: Comprehensive technical analysis
  - `strategic`: Focus on strategic implications
  - `comprehensive`: Complete analysis with all sections (default)

- **Output Formats**:
  - `json`: Structured data (default)
  - `csv`: Tabular format for spreadsheets
  - `html`: Rich HTML report with visualizations

### Interactive Session

The interactive mode guides you through:
1. LLM configuration
2. Query input
3. Analysis parameter selection
4. Real-time progress tracking
5. Results summary and export options

## 🏗️ Architecture

### Model Context Protocol (MCP) Implementation

```
┌─────────────────────────────────────────────────────────────┐
│                    TrendRadarOrchestrator                   │
│                     (MCP Coordinator)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────┐ │
│  │DataCollector│◄─┤AnalysisAgent├─►│Visualization│◄─┤Report│ │
│  │   Agent     │  │             │  │   Agent     │  │Agent│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────┘ │
├─────────────────────────────────────────────────────────────┤
│                    LLM Client (gpt-oss:20b)                │
│                       via Ollama                           │
└─────────────────────────────────────────────────────────────┘
```

### Agent Responsibilities

#### 🔍 DataCollectorAgent
- Trend signal detection
- Multi-source data aggregation
- Data quality assessment
- Source credibility validation

#### 📊 AnalysisAgent  
- Impact scoring (1-4 scale)
- Confidence assessment (0.0-1.0)
- Timeline prediction
- Risk/opportunity analysis
- Cross-trend correlation

#### 📈 VisualizationAgent
- Radar chart generation
- Supporting visualizations
- Statistical summaries
- Export format creation
- Interactive dashboard configs

#### 📋 ReportingAgent
- Executive summaries
- Strategic insights extraction
- Recommendation generation
- Detailed analysis reports
- Multi-format outputs

## 🛠️ Development

### Project Structure

```
trend-radar-mcp/
├── pyproject.toml              # uv package configuration
├── README.md                   # This file
├── src/trend_radar/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── models/                 # Data models
│   │   ├── trend.py           # Trend data structures
│   │   └── mcp_message.py     # MCP protocol messages
│   ├── agents/                 # MCP agents
│   │   ├── base_agent.py      # Base agent class
│   │   ├── data_collector.py  # Data collection agent
│   │   ├── analysis_agent.py  # Analysis agent
│   │   ├── visualization_agent.py # Visualization agent
│   │   └── reporting_agent.py # Reporting agent
│   ├── orchestrator/          # Main coordination
│   │   └── trend_radar_orchestrator.py
│   └── utils/                 # Utilities
│       ├── logger.py          # Logging configuration
│       └── llm_client.py      # LLM communication
└── tests/                     # Test suite
    └── test_agents.py
```

### Development Setup

1. **Install development dependencies**:
   ```bash
   uv sync --dev
   ```

2. **Run tests**:
   ```bash
   uv run pytest
   ```

3. **Code formatting**:
   ```bash
   uv run black src/
   uv run ruff check src/
   ```

4. **Type checking**:
   ```bash
   uv run mypy src/
   ```

### Adding New Agents

1. Inherit from `MCPAgent` base class
2. Implement `_define_capabilities()` method
3. Implement `process_task()` method
4. Register with orchestrator
5. Add to agent health checks

Example:
```python
class CustomAgent(MCPAgent):
    def _define_capabilities(self) -> List[AgentCapability]:
        return [AgentCapability(
            name="custom_analysis",
            description="Custom trend analysis",
            input_types=["trend_data"],
            output_types=["analysis_results"],
            confidence_level=0.8
        )]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Implement custom logic
        return {"result": "custom analysis completed"}
```

## 🔧 Configuration

### LLM Configuration

The application supports various LLM configurations:

```python
llm_config = {
    "base_url": "http://localhost:11434",  # Ollama endpoint
    "model": "gpt-oss:20b",               # Model name
    "timeout": 60                         # Request timeout
}
```

### Analysis Parameters

```python
analysis_config = {
    "depth": "standard",                  # Analysis depth
    "focus_areas": ["impact", "confidence", "timeline"],
    "report_type": "comprehensive",       # Report format
    "target_audience": "leadership"       # Report audience
}
```

## 📊 Output Examples

### Trend Radar Visualization Data

```json
{
  "radar_data": [
    {
      "id": "trend_1_20250113",
      "title": "AI-Powered Code Generation",
      "category": "technology",
      "x": 2,              // Time horizon (1-4 scale)
      "y": 3,              // Impact level (1-4 scale)
      "size": 18,          // Confidence-based size
      "confidence": 0.85,
      "impact_level": "high",
      "time_horizon_label": "Short-term"
    }
  ]
}
```

### Strategic Recommendations

```json
{
  "strategic_recommendations": [
    {
      "title": "Invest in AI Development Capabilities",
      "description": "Build internal AI expertise to leverage emerging opportunities",
      "priority": "high",
      "timeframe": "6-12 months",
      "effort": "medium",
      "impact": "strategic"
    }
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and formatting (`uv run pytest && uv run black src/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM hosting
- **uv** for modern Python package management
- **Rich** for beautiful terminal output
- **Typer** for CLI interface
- **Model Context Protocol** specification

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   # Try running from project root directory
   cd trend-radar-mcp
   python -m trend_radar.main analyze "your query"
   
   # Or use the quick start script
   ./run_trend_analysis.py analyze "your query"
   
   # Or install dependencies
   uv sync
   ```

2. **"Connection refused" errors**:
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Check if gpt-oss:20b is pulled
   ollama list
   
   # Pull the model if missing
   ollama pull gpt-oss:20b
   ```

3. **"Model not found" errors**:
   ```bash
   # Pull the model
   ollama pull gpt-oss:20b
   
   # Verify model name in configuration
   python -m trend_radar.main analyze "test" --model gpt-oss:20b
   ```

4. **"Command not found: trend-radar"**:
   ```bash
   # Install the package first
   uv pip install -e .
   
   # Or use direct execution methods
   python -m trend_radar.main analyze "your query"
   
   # Or use the quick start script
   ./run_trend_analysis.py analyze "your query"
   ```

5. **Permission errors with run script**:
   ```bash
   # Make the script executable
   chmod +x run_trend_analysis.py
   
   # Or run with python directly
   python run_trend_analysis.py analyze "your query"
   ```

6. **Slow performance**:
   ```bash
   # Reduce analysis depth to "light"
   python -m trend_radar.main analyze "query" --depth light
   
   # Check system resources (RAM, CPU)
   # Ensure adequate GPU/CPU for model
   ```

7. **Import errors**:
   ```bash
   # Reinstall dependencies
   uv sync --reinstall
   
   # Check Python version compatibility (>=3.8)
   python --version
   
   # Try running from correct directory
   cd trend-radar-mcp
   ```

### Debug Mode

Enable verbose logging:
```bash
# Set environment variable for debug logging
export PYTHONPATH=src
python -c "
from trend_radar.utils.logger import configure_root_logger
configure_root_logger('DEBUG', use_rich=True)
import asyncio
from trend_radar.main import run_trend_analysis
asyncio.run(run_trend_analysis('debug test', {'base_url': 'http://localhost:11434', 'model': 'gpt-oss:20b'}, {'depth': 'light'}))
"
```

### Test the Installation

```bash
# Quick test to verify everything works
./run_trend_analysis.py health-check

# Or
python -m trend_radar.main health-check

# Run the demo to test full functionality
./run_trend_analysis.py demo

# Or
python -m trend_radar.main demo
```

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/your-org/trend-radar-mcp/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-org/trend-radar-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/trend-radar-mcp/discussions)

---

**Made with ❤️ using Model Context Protocol and AI Agent Orchestration**
