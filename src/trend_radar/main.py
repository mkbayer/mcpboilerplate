"""
Main application entry point for the Trend Radar MCP system.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from .orchestrator.trend_radar_orchestrator import TrendRadarOrchestrator
from .utils.logger import configure_root_logger, get_logger

# Initialize console and logger
console = Console()
configure_root_logger("INFO", use_rich=True)
logger = get_logger(__name__)

# Create Typer app
app = typer.Typer(
    name="trend-radar",
    help="🎯 Trend Radar MCP Application - AI-powered trend analysis with multi-agent orchestration",
    rich_markup_mode="rich"
)

# Global configuration
DEFAULT_LLM_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "gpt-oss:20b"
}


@app.command()
def analyze(
    query: str = typer.Argument("emerging technology trends 2025", help="Query or topic for trend analysis"),
    depth: str = typer.Option("standard", help="Analysis depth: light, standard, or deep"),
    output_format: str = typer.Option("json", help="Output format: json, csv, or html"),
    output_file: Optional[str] = typer.Option(None, help="Output filename (auto-generated if not specified)"),
    interactive: bool = typer.Option(True, help="Run in interactive mode with progress display"),
    report_type: str = typer.Option("comprehensive", help="Report type: executive, detailed, strategic, or comprehensive"),
    llm_url: str = typer.Option("http://localhost:11434", help="Ollama base URL"),
    model: str = typer.Option("gpt-oss:20b", help="LLM model name")
):
    """
    🔍 Run comprehensive trend analysis using MCP agent orchestration.
    
    This command orchestrates four specialized AI agents to collect, analyze, 
    visualize, and report on trends for the given query.
    """
    console.print(Panel.fit("🎯 [bold blue]Trend Radar MCP Application[/bold blue]", border_style="blue"))
    
    # Prepare configuration
    llm_config = {
        "base_url": llm_url,
        "model": model
    }
    
    analysis_config = {
        "depth": depth,
        "report_type": report_type,
        "target_audience": "leadership"
    }
    
    # Run the analysis
    results = asyncio.run(
        run_trend_analysis(
            query=query,
            llm_config=llm_config,
            analysis_config=analysis_config,
            interactive=interactive,
            output_format=output_format,
            output_file=output_file
        )
    )
    
    if results.get("processing_complete"):
        console.print("\n✅ [green]Analysis completed successfully![/green]")
    else:
        console.print(f"\n❌ [red]Analysis failed: {results.get('error', 'Unknown error')}[/red]")
        raise typer.Exit(1)


@app.command()
def health_check(
    llm_url: str = typer.Option("http://localhost:11434", help="Ollama base URL"),
    model: str = typer.Option("gpt-oss:20b", help="LLM model name"),
    verbose: bool = typer.Option(False, help="Show detailed health information")
):
    """
    🏥 Check health status of all MCP agents and LLM connectivity.
    """
    console.print("🏥 [bold yellow]Performing Health Check...[/bold yellow]")
    
    llm_config = { "model": model}
    
    health_status = asyncio.run(check_system_health(llm_config, verbose))
    
    if health_status["overall_healthy"]:
        console.print("\n✅ [green]All systems operational![/green]")
    else:
        console.print(f"\n❌ [red]Health check failed: {health_status['issues']}[/red]")
        raise typer.Exit(1)


@app.command()
def interactive():
    """
    🎮 Launch interactive trend analysis session.
    """
    console.print(Panel.fit("🎮 [bold green]Interactive Trend Radar Session[/bold green]", border_style="green"))
    
    asyncio.run(run_interactive_session())


@app.command()  
def demo():
    """
    🚀 Run a demonstration with sample trend analysis.
    """
    console.print(Panel.fit("🚀 [bold magenta]Trend Radar Demo[/bold magenta]", border_style="magenta"))
    
    demo_queries = [
        "artificial intelligence trends 2025",
        "sustainable technology innovations",
        "future of remote work and collaboration",
    ]
    
    console.print("Running demonstration with sample queries...")
    
    for i, query in enumerate(demo_queries, 1):
        console.print(f"\n[bold cyan]Demo {i}/3: {query}[/bold cyan]")
        
        results = asyncio.run(
            run_trend_analysis(
                query=query,
                llm_config=DEFAULT_LLM_CONFIG,
                analysis_config={"depth": "light", "report_type": "executive"},
                interactive=True,
                output_format="json",
                output_file=f"demo_results_{i}"
            )
        )
        
        if results.get("processing_complete"):
            console.print(f"✅ Demo {i} completed")
        else:
            console.print(f"❌ Demo {i} failed")


async def run_trend_analysis(
    query: str,
    llm_config: Dict[str, Any],
    analysis_config: Dict[str, Any],
    interactive: bool = True,
    output_format: str = "json",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the complete trend analysis pipeline
    
    Args:
        query: Analysis query
        llm_config: LLM configuration
        analysis_config: Analysis configuration
        interactive: Whether to show interactive progress
        output_format: Export format
        output_file: Output filename
        
    Returns:
        Analysis results dictionary
    """
    orchestrator = TrendRadarOrchestrator(llm_config)
    
    if interactive:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            # Add progress task
            task = progress.add_task(f"Analyzing: {query[:50]}...", total=None)
            
            try:
                # Run analysis
                results = await orchestrator.orchestrate_trend_analysis(query, analysis_config)
                
                progress.update(task, description="✅ Analysis completed")
                progress.stop()
                
                # Display results summary
                if results.get("processing_complete"):
                    display_results_summary(results)
                    
                    # Export results
                    if output_format and output_format != "none":
                        export_path = await orchestrator.export_results(
                            results, output_format, output_file
                        )
                        console.print(f"📁 Results exported to: [green]{export_path}[/green]")
                
                return results
                
            except Exception as e:
                progress.update(task, description=f"❌ Failed: {str(e)}")
                progress.stop()
                raise
                
            finally:
                await orchestrator.cleanup_session()
    
    else:
        # Non-interactive mode
        try:
            results = await orchestrator.orchestrate_trend_analysis(query, analysis_config)
            
            if output_format and output_format != "none":
                await orchestrator.export_results(results, output_format, output_file)
            
            return results
            
        finally:
            await orchestrator.cleanup_session()


def display_results_summary(results: Dict[str, Any]) -> None:
    """Display a summary of analysis results"""
    
    # Extract key metrics
    pipeline_summary = results.get("pipeline_summary", {})
    radar_stats = results.get("trend_radar", {}).get("statistics", {}).get("overview", {})
    insights = results.get("report", {}).get("key_insights", [])
    recommendations = results.get("report", {}).get("strategic_recommendations", [])
    
    # Create summary table
    table = Table(title="📊 Analysis Summary", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Trends Processed", str(pipeline_summary.get("total_trends_processed", 0)))
    table.add_row("Average Confidence", f"{radar_stats.get('average_confidence', 0):.1%}")
    table.add_row("Average Impact", f"{radar_stats.get('average_impact', 0):.1f}/4")
    table.add_row("Key Insights", str(len(insights)))
    table.add_row("Recommendations", str(len(recommendations)))
    
    console.print(table)
    
    # Display top insights
    if insights:
        console.print("\n💡 [bold yellow]Top Insights:[/bold yellow]")
        for i, insight in enumerate(insights[:3], 1):
            console.print(f"  {i}. [bold]{insight.get('title', 'Unknown')}[/bold]")
            console.print(f"     {insight.get('description', 'No description')[:100]}...")
    
    # Display top recommendations  
    if recommendations:
        console.print("\n🎯 [bold green]Key Recommendations:[/bold green]")
        for i, rec in enumerate(recommendations[:3], 1):
            priority_color = {"high": "red", "medium": "yellow", "low": "blue"}.get(rec.get('priority', 'medium'), "white")
            console.print(f"  {i}. [{priority_color}]{rec.get('priority', 'medium').upper()}[/{priority_color}] [bold]{rec.get('title', 'Unknown')}[/bold]")
            console.print(f"     {rec.get('description', 'No description')[:100]}...")


async def check_system_health(llm_config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Check health of all system components"""
    
    orchestrator = TrendRadarOrchestrator(llm_config)
    health_status = {"overall_healthy": True, "components": {}, "issues": []}
    
    try:
        # Check agent status
        agent_status = await orchestrator.get_agent_status()
        
        for agent_id, status in agent_status["agents"].items():
            is_healthy = status.get("health_status") == "healthy"
            health_status["components"][agent_id] = {
                "healthy": is_healthy,
                "status": status.get("status", "unknown")
            }
            
            if not is_healthy:
                health_status["overall_healthy"] = False
                health_status["issues"].append(f"Agent {agent_id} is unhealthy")
        
        if verbose:
            console.print("🔍 [bold]Detailed Health Status:[/bold]")
            
            # Create health table
            health_table = Table(title="Agent Health Status", show_header=True)
            health_table.add_column("Agent", style="cyan")
            health_table.add_column("Status", style="green")
            health_table.add_column("Health", style="yellow")
            
            for agent_id, component in health_status["components"].items():
                status_icon = "✅" if component["healthy"] else "❌"
                health_table.add_row(
                    agent_id,
                    component["status"], 
                    f"{status_icon} {'Healthy' if component['healthy'] else 'Unhealthy'}"
                )
            
            console.print(health_table)
        
        return health_status
        
    except Exception as e:
        health_status["overall_healthy"] = False
        health_status["issues"].append(f"System error: {str(e)}")
        return health_status
    
    finally:
        await orchestrator.cleanup_session()


async def run_interactive_session():
    """Run an interactive trend analysis session"""
    
    console.print("Welcome to the interactive Trend Radar session!")
    console.print("You can analyze multiple trends and customize the analysis parameters.\n")
    
    # Session configuration
    session_config = {}
    
    # Get LLM configuration
    console.print("🔧 [bold]LLM Configuration[/bold]")
    llm_url = Prompt.ask("Ollama URL", default="http://localhost:11434")
    model = Prompt.ask("Model name", default="gpt-oss:20b")
    
    session_config["llm"] = {"base_url": llm_url, "model": model}
    
    # Analysis loop
    session_results = []
    
    while True:
        console.print("\n" + "="*50)
        console.print("🎯 [bold cyan]New Trend Analysis[/bold cyan]")
        
        # Get query
        query = Prompt.ask("Enter your trend analysis query")
        if not query.strip():
            console.print("Empty query. Exiting...")
            break
        
        # Get analysis parameters
        console.print("\n📋 [bold]Analysis Configuration[/bold]")
        depth = Prompt.ask("Analysis depth", choices=["light", "standard", "deep"], default="standard")
        report_type = Prompt.ask("Report type", choices=["executive", "detailed", "strategic", "comprehensive"], default="comprehensive")
        
        analysis_config = {
            "depth": depth,
            "report_type": report_type,
            "target_audience": "leadership"
        }
        
        # Run analysis
        try:
            results = await run_trend_analysis(
                query=query,
                llm_config=session_config["llm"],
                analysis_config=analysis_config,
                interactive=True,
                output_format="json",
                output_file=None
            )
            
            session_results.append({
                "query": query,
                "timestamp": results.get("execution_metadata", {}).get("completed_at", "unknown"),
                "success": results.get("processing_complete", False)
            })
            
        except Exception as e:
            console.print(f"❌ Analysis failed: {str(e)}")
            session_results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
        
        # Ask if user wants to continue
        if not Confirm.ask("\nAnalyze another trend?"):
            break
    
    # Session summary
    console.print("\n📋 [bold blue]Session Summary[/bold blue]")
    
    if session_results:
        summary_table = Table(show_header=True, header_style="bold blue")
        summary_table.add_column("Query", style="cyan")
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Timestamp", style="yellow")
        
        for result in session_results:
            status_icon = "✅" if result["success"] else "❌"
            summary_table.add_row(
                result["query"][:40] + "..." if len(result["query"]) > 40 else result["query"],
                f"{status_icon} {'Success' if result['success'] else 'Failed'}",
                result.get("timestamp", "unknown")[:16]  # Show date and time only
            )
        
        console.print(summary_table)
        
        successful_analyses = sum(1 for r in session_results if r["success"])
        console.print(f"\nCompleted {successful_analyses}/{len(session_results)} analyses successfully")
    
    console.print("\n👋 Session ended. Thank you for using Trend Radar!")


def main():
    """Main entry point for the application"""
    
    # Check if we're being run directly with no arguments
    if len(sys.argv) == 1:
        # No arguments provided, show help
        console.print(Panel.fit(
            "🎯 [bold blue]Trend Radar MCP Application[/bold blue]\n\n" +
            "Available commands:\n" +
            "• [green]analyze[/green] - Run trend analysis\n" +
            "• [green]interactive[/green] - Launch interactive session\n" +
            "• [green]health-check[/green] - Check system health\n" +
            "• [green]demo[/green] - Run demonstration\n\n" +
            "Use --help with any command for more details\n" +
            "Example: [cyan]python -m trend_radar.main analyze \"AI trends\"[/cyan]",
            border_style="blue"
        ))
        console.print("\n💡 [yellow]Tip: Run 'python -m trend_radar.main interactive' for guided analysis[/yellow]")
        return
    
    # Run the Typer app
    app()


if __name__ == "__main__":
    main()
    