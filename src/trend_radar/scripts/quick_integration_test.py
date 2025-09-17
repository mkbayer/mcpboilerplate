#!/usr/bin/env python3
"""
Quick test to verify the HTML integration actually works.
"""

import sys
import os
from pathlib import Path
import base64

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
os.chdir(project_root)

def test_base64_embedding_simple():
    print("🔍 Testing base64 image embedding...")
    
    try:
        # Create a simple test image
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([1, 2, 3, 4], [2, 3, 1, 4], s=[50, 100, 75, 125], alpha=0.7)
        ax.set_title('Test Plot for HTML Embedding')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        
        test_plot_path = Path("test_embed.png")
        plt.savefig(test_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if not test_plot_path.exists():
            print("❌ Test plot was not created")
            return False
        
        print(f"✅ Test plot created: {test_plot_path} ({test_plot_path.stat().st_size} bytes)")
        
        # Convert to base64
        with open(test_plot_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            base64_image = f"data:image/png;base64,{image_data}"
        
        print(f"✅ Base64 encoding successful: {len(base64_image)} characters")
        
        # Create test HTML with embedded image
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test HTML with Embedded Plot</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .plot-container {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>🧪 Test HTML Report with Embedded Plot</h1>
            <p>This is a test to verify that plots can be embedded in HTML reports.</p>
            
            <div class="plot-container">
                <h3>📊 Test Scatter Plot</h3>
                <img src="{base64_image}" alt="Test Plot" />
                <p><em>This plot is embedded as a base64 image</em></p>
            </div>
            
            <p>✅ If you can see the plot above, the HTML integration is working!</p>
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = Path("test_embedded_plot.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Clean up test plot file
        test_plot_path.unlink()
        
        if html_file.exists():
            html_size = html_file.stat().st_size
            print(f"✅ HTML file created: {html_file} ({html_size} bytes)")
            print(f"🌐 Open with: open {html_file} (Mac) or start {html_file} (Windows)")
            print("💡 Check if you can see the embedded plot in the HTML file")
            return True
        else:
            print("❌ HTML file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_embedding_method():
    print("\n🔍 Testing orchestrator embedding method...")
    
    try:
        from trend_radar.orchestrator.trend_radar_orchestrator import TrendRadarOrchestrator
        
        orchestrator = TrendRadarOrchestrator()
        
        # Check if the method exists
        if hasattr(orchestrator, '_embed_plot_images'):
            print("✅ _embed_plot_images method found in orchestrator")
            
            # Create a test plot for embedding
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots()
            ax.bar(['A', 'B', 'C'], [1, 3, 2])
            ax.set_title('Test Bar Chart')
            
            test_plot = Path("orchestrator_test.png")
            plt.savefig(test_plot)
            plt.close()
            
            # Test the embedding method
            plot_files = {"test_bar": str(test_plot)}
            embedded = orchestrator._embed_plot_images(plot_files, {})
            
            if "test_bar" in embedded and embedded["test_bar"]:
                if embedded["test_bar"].startswith("data:image/png;base64,"):
                    print("✅ Orchestrator embedding method works correctly")
                    test_plot.unlink()  # Clean up
                    return True
                else:
                    print("❌ Embedded data format is incorrect")
            else:
                print("❌ No embedded data returned")
            
            test_plot.unlink()  # Clean up
            return False
            
        else:
            print("❌ _embed_plot_images method not found in orchestrator")
            print("🔍 Available methods:")
            methods = [method for method in dir(orchestrator) if not method.startswith('_') or method.startswith('_embed')]
            for method in methods:
                print(f"   • {method}")
            return False
            
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Quick HTML Integration Test\n")
    
    # Test 1: Basic base64 embedding
    basic_success = test_base64_embedding_simple()
    
    # Test 2: Orchestrator method
    orchestrator_success = test_orchestrator_embedding_method()
    
    print(f"\n📊 Test Results:")
    print(f"   Basic embedding: {'✅ Success' if basic_success else '❌ Failed'}")
    print(f"   Orchestrator method: {'✅ Success' if orchestrator_success else '❌ Failed'}")
    
    if basic_success and orchestrator_success:
        print("\n🎉 HTML integration should work! Both tests passed.")
        print("💡 Try running: ./run.sh generate-report \"AI trends\"")
    elif basic_success:
        print("\n⚠️  Basic embedding works, but orchestrator method has issues.")
        print("🔧 The _embed_plot_images method may need to be added to the orchestrator.")
    else:
        print("\n❌ Basic embedding failed. Check matplotlib installation.")

if __name__ == "__main__":
    main()
    