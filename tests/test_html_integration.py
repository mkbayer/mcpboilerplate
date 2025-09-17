#!/usr/bin/env python3
"""
Test script to verify HTML report with embedded plots works end-to-end.
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

async def test_html_integration_end_to_end():
    print("🔍 Testing HTML report with embedded plots integration...\n")
    
    # Step 1: Run a complete analysis
    print("1. Running complete trend analysis...")
    try:
        from src.trend_radar.orchestrator import TrendRadarOrchestrator
        
        orchestrator = TrendRadarOrchestrator()
        
        # Run analysis with sample query
        config = {
            "depth": "light",
            "report_type": "comprehensive",
            "include_plots": True
        }
        
        print("   Executing orchestration pipeline...")
        results = await orchestrator.orchestrate_trend_analysis(
            "AI trends test", config
        )
        
        if not results.get("processing_complete"):
            print(f"   ❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return False
        
        print("   ✅ Analysis completed successfully")
        
        # Check if plots were created
        plot_files = results.get("trend_radar", {}).get("plot_files", {})
        supporting_files = results.get("trend_radar", {}).get("supporting_plot_files", {})
        
        total_plots = len(plot_files) + len(supporting_files)
        print(f"   📊 Plots generated: {total_plots}")
        
        if total_plots == 0:
            print("   ❌ No plots were generated!")
            return False
        
        # List generated plots
        print("   Generated plot files:")
        for name, path in plot_files.items():
            exists = Path(path).exists() if path else False
            size = Path(path).stat().st_size if exists else 0
            print(f"     • {name}: {path} {'✅' if exists else '❌'} ({size} bytes)")
        
        for name, path in supporting_files.items():
            exists = Path(path).exists() if path else False
            size = Path(path).stat().st_size if exists else 0
            print(f"     • {name}: {path} {'✅' if exists else '❌'} ({size} bytes)")
        
    except Exception as e:
        print(f"   ❌ Analysis failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Test HTML export with embedded plots
    print("\n2. Testing HTML export with embedded plots...")
    try:
        # Generate HTML report
        html_path = "test_report_with_plots.html"
        await orchestrator.export_results(results, "html", "test_report_with_plots")
        
        html_file = Path(f"{html_path}")
        if html_file.exists():
            file_size = html_file.stat().st_size
            print(f"   ✅ HTML report created: {html_file} ({file_size} bytes)")
            
            # Check if HTML contains embedded images
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Look for base64 embedded images
            base64_images = html_content.count('data:image/png;base64,')
            print(f"   🖼️  Embedded images found: {base64_images}")
            
            if base64_images > 0:
                print("   ✅ Plots are successfully embedded in HTML!")
                
                # Check HTML structure
                has_plots_section = '<div class="section plots-section">' in html_content
                has_radar_plot = 'Main Trend Radar Chart' in html_content
                has_metrics = 'Key Metrics' in html_content
                has_insights = 'Key Insights' in html_content
                
                print(f"   📋 HTML sections:")
                print(f"     • Plots section: {'✅' if has_plots_section else '❌'}")
                print(f"     • Radar chart: {'✅' if has_radar_plot else '❌'}")
                print(f"     • Metrics: {'✅' if has_metrics else '❌'}")
                print(f"     • Insights: {'✅' if has_insights else '❌'}")
                
                # Estimate total embedded image size
                base64_start = html_content.find('data:image/png;base64,')
                if base64_start != -1:
                    # Count approximate base64 data
                    remaining_content = html_content[base64_start:]
                    base64_data_size = 0
                    for line in remaining_content.split('"'):
                        if line.startswith('data:image/png;base64,'):
                            base64_data_size += len(line)
                    
                    estimated_image_size = (base64_data_size * 3 // 4) // 1024  # Convert base64 to approximate KB
                    print(f"   📊 Estimated embedded image size: ~{estimated_image_size} KB")
                
                return True
            else:
                print("   ❌ No embedded images found in HTML!")
                print("   🔍 Checking HTML content preview:")
                print(f"     HTML length: {len(html_content)} characters")
                print(f"     Contains img tags: {'<img' in html_content}")
                print(f"     Contains plot-container: {'plot-container' in html_content}")
                return False
        else:
            print(f"   ❌ HTML file was not created: {html_file}")
            return False
            
    except Exception as e:
        print(f"   ❌ HTML export failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            await orchestrator.cleanup_session()
        except:
            pass

def test_base64_embedding():
    """Test the base64 embedding function directly"""
    print("\n3. Testing base64 embedding function directly...")
    
    try:
        from src.trend_radar.orchestrator import TrendRadarOrchestrator
        
        orchestrator = TrendRadarOrchestrator()
        
        # Create a test PNG file
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title('Test Plot for Embedding')
        
        test_plot_path = "test_embedding_plot.png"
        plt.savefig(test_plot_path)
        plt.close()
        
        if not Path(test_plot_path).exists():
            print("   ❌ Test plot file was not created")
            return False
        
        print(f"   ✅ Test plot created: {test_plot_path}")
        
        # Test embedding
        plot_files = {"test_plot": test_plot_path}
        embedded = orchestrator._embed_plot_images(plot_files, {})
        
        if "test_plot" in embedded and embedded["test_plot"]:
            base64_data = embedded["test_plot"]
            if base64_data.startswith("data:image/png;base64,"):
                print("   ✅ Base64 embedding works correctly")
                print(f"   📏 Base64 data length: {len(base64_data)} characters")
                
                # Clean up test file
                Path(test_plot_path).unlink()
                return True
            else:
                print("   ❌ Base64 data format is incorrect")
        else:
            print("   ❌ Embedding function returned no data")
        
        # Clean up test file
        Path(test_plot_path).unlink()
        return False
        
    except Exception as e:
        print(f"   ❌ Base64 embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🧪 Testing HTML Report with Embedded Plots\n")
    
    # Run end-to-end test
    integration_success = await test_html_integration_end_to_end()
    
    # Test base64 embedding directly
    embedding_success = test_base64_embedding()
    
    print(f"\n📊 Test Results:")
    print(f"   End-to-end integration: {'✅ Success' if integration_success else '❌ Failed'}")
    print(f"   Base64 embedding: {'✅ Success' if embedding_success else '❌ Failed'}")
    
    if integration_success and embedding_success:
        print("\n🎉 All tests passed! HTML integration with embedded plots works correctly.")
        print("💡 You can now use: ./run.sh generate-report \"your query\" to create beautiful reports")
        
        # Show the test report
        test_report = Path("test_report_with_plots.html")
        if test_report.exists():
            print(f"\n📄 Test report created: {test_report.absolute()}")
            print(f"🌐 Open with: open {test_report} (Mac) or start {test_report} (Windows)")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure all dependencies are installed: pip install matplotlib plotly pandas")
        print("   2. Check if the orchestrator can create plots")
        print("   3. Verify file permissions for HTML creation")

if __name__ == "__main__":
    asyncio.run(main())
    