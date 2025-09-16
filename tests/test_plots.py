#!/usr/bin/env python3
"""
Test script to verify plot generation works correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
os.chdir(project_root)

def test_plot_generation():
    print("🔍 Testing plot generation...")
    
    try:
        # Test matplotlib availability
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        # Test basic plot creation
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter([1, 2, 3, 4], [2, 3, 1, 4], s=[50, 100, 75, 125], alpha=0.7)
        ax.set_title('Test Plot')
        
        # Save test plot
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        test_plot_path = plots_dir / "test_plot.png"
        
        plt.savefig(test_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if test_plot_path.exists():
            file_size = test_plot_path.stat().st_size
            print(f"✅ Test plot created: {test_plot_path} ({file_size} bytes)")
            
            # Clean up test file
            test_plot_path.unlink()
            print("🧹 Test plot cleaned up")
            
            return True
        else:
            print("❌ Test plot file was not created")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install matplotlib: pip install matplotlib")
        return False
    except Exception as e:
        print(f"❌ Plot generation failed: {e}")
        return False

def test_plotter_class():
    print("\n🔍 Testing TrendRadarPlotter class...")
    
    try:
        from trend_radar.utils.plotter import TrendRadarPlotter
        print("✅ TrendRadarPlotter imported successfully")
        
        plotter = TrendRadarPlotter()
        print(f"✅ TrendRadarPlotter initialized: {plotter.output_dir}")
        
        # Test with sample data
        sample_radar_data = [
            {
                'id': 'test_1',
                'title': 'AI Development Trends',
                'category': 'technology',
                'x': 2,
                'y': 3,
                'size': 100,
                'confidence': 0.8,
                'description': 'Test trend for AI development'
            },
            {
                'id': 'test_2', 
                'title': 'Business Innovation',
                'category': 'business',
                'x': 3,
                'y': 2,
                'size': 80,
                'confidence': 0.6,
                'description': 'Test trend for business innovation'
            }
        ]
        
        # Create test plot
        plot_path = plotter.create_trend_radar(sample_radar_data, "Test Radar Chart")
        
        if plot_path and Path(plot_path).exists():
            file_size = Path(plot_path).stat().st_size
            print(f"✅ Test radar plot created: {plot_path} ({file_size} bytes)")
            
            # List all files in plots directory
            plots_dir = Path("plots")
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*"))
                print(f"📁 Plot directory contains {len(plot_files)} files:")
                for file in plot_files:
                    print(f"   • {file.name} ({file.stat().st_size} bytes)")
            
            return True
        else:
            print("❌ Test radar plot was not created")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Check if all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ TrendRadarPlotter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🎯 Testing Visualization Components\n")
    
    # Test basic matplotlib
    matplotlib_ok = test_plot_generation()
    
    # Test plotter class
    plotter_ok = test_plotter_class()
    
    print(f"\n📊 Test Results:")
    print(f"   Matplotlib: {'✅' if matplotlib_ok else '❌'}")
    print(f"   TrendRadarPlotter: {'✅' if plotter_ok else '❌'}")
    
    if matplotlib_ok and plotter_ok:
        print("\n🎉 All visualization tests passed!")
        print("💡 You can now run: ./run.sh analyze \"test query\" to generate real plots")
    else:
        print("\n❌ Some tests failed. Check error messages above.")
        
        print("\n🔧 Troubleshooting:")
        print("   1. Install missing dependencies: pip install matplotlib seaborn plotly pandas")
        print("   2. Ensure you're in the project root directory")
        print("   3. Check that src/trend_radar/ directory exists")

if __name__ == "__main__":
    main()
    