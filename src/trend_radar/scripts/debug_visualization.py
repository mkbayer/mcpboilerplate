#!/usr/bin/env python3
"""
Debug script to test visualization step by step
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
os.chdir(project_root)

async def debug_visualization_step_by_step():
    print("🔍 Debugging visualization step by step...\n")
    
    # Step 1: Test basic matplotlib
    print("1. Testing matplotlib...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create simple test plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title('Basic Test')
        
        test_path = Path("test_matplotlib.png")
        plt.savefig(test_path)
        plt.close()
        
        if test_path.exists():
            print(f"   ✅ Matplotlib works: {test_path} ({test_path.stat().st_size} bytes)")
            test_path.unlink()  # Clean up
        else:
            print("   ❌ Matplotlib failed to create file")
        
    except Exception as e:
        print(f"   ❌ Matplotlib error: {e}")
        return False
    
    # Step 2: Test TrendRadarPlotter
    print("\n2. Testing TrendRadarPlotter...")
    try:
        from  ..utils.plotter import TrendRadarPlotter
        
        plotter = TrendRadarPlotter()
        print(f"   ✅ Plotter created, output dir: {plotter.output_dir.absolute()}")
        
        # Create sample data
        sample_data = [
            {
                'id': 'test1',
                'title': 'AI Trends',
                'category': 'technology',
                'x': 2.0,
                'y': 3.0,
                'size': 100.0,
                'confidence': 0.8,
                'description': 'Test AI trend'
            }
        ]
        
        # Try to create plot
        print("   Creating test radar plot...")
        plot_path = plotter.create_trend_radar(sample_data, "Test Radar")
        
        if plot_path and Path(plot_path).exists():
            file_size = Path(plot_path).stat().st_size
            print(f"   ✅ Radar plot created: {plot_path} ({file_size} bytes)")
        else:
            print(f"   ❌ Radar plot failed: {plot_path}")
            return False
            
    except Exception as e:
        print(f"   ❌ TrendRadarPlotter error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test VisualizationAgent
    print("\n3. Testing VisualizationAgent...")
    try:
        from trend_radar.agents.visualization_agent import VisualizationAgent
        
        viz_agent = VisualizationAgent()
        print(f"   ✅ VisualizationAgent created: {viz_agent.agent_id}")
        
        # Create sample analyzed trends
        sample_trends = [
            {
                'id': 'trend1',
                'title': 'Machine Learning Advances',
                'description': 'New developments in ML algorithms',
                'category': 'technology',
                'confidence': 0.8,
                'impact': 'high',
                'time_horizon': 'short_term'
            },
            {
                'id': 'trend2',
                'title': 'Remote Work Evolution',
                'description': 'Changes in remote work practices',
                'category': 'business', 
                'confidence': 0.7,
                'impact': 'medium',
                'time_horizon': 'medium_term'
            }
        ]
        
        # Test process_task
        task = {
            "analyzed_trends": sample_trends,
            "type": "radar",
            "config": {"title": "Debug Test Radar"}
        }
        
        print("   Processing visualization task...")
        result = await viz_agent.process_task(task)
        
        print(f"   ✅ Task completed. Processing complete: {result.get('metadata', {}).get('plots_created', 0)} plots")
        
        # Check results
        plot_files = result.get("plot_files", {})
        supporting_files = result.get("supporting_plot_files", {})
        
        print("   Plot files returned:")
        for name, path in plot_files.items():
            exists = Path(path).exists() if path else False
            print(f"     {name}: {path} {'✅' if exists else '❌'}")
        
        print("   Supporting files returned:")
        for name, path in supporting_files.items():
            exists = Path(path).exists() if path else False
            print(f"     {name}: {path} {'✅' if exists else '❌'}")
        
        return len(plot_files) > 0 or len(supporting_files) > 0
        
    except Exception as e:
        print(f"   ❌ VisualizationAgent error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Check plots directory
    print("\n4. Checking plots directory...")
    plots_dir = Path("plots")
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*"))
        print(f"   📁 Plots directory: {plots_dir.absolute()}")
        print(f"   📊 Files found: {len(plot_files)}")
        
        for file in plot_files:
            size = file.stat().st_size
            print(f"     • {file.name} ({size} bytes)")
    else:
        print(f"   ❌ Plots directory does not exist: {plots_dir.absolute()}")

if __name__ == "__main__":
    print("🎯 Debugging Visualization Pipeline\n")
    success = asyncio.run(debug_visualization_step_by_step())
    
    print(f"\n📊 Debug Results: {'✅ Success' if success else '❌ Failed'}")
    
    if not success:
        print("\n🔧 Troubleshooting steps:")
        print("1. Install dependencies: pip install matplotlib seaborn plotly pandas")
        print("2. Check file permissions in current directory")
        print("3. Try running from project root directory")
        print("4. Check if display/GUI is available (for matplotlib backend)")
        