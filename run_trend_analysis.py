#!/usr/bin/env python3
"""
Quick start script for Trend Radar MCP Application
Run this script directly without installation.
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import trend_radar
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project directory to ensure relative imports work
os.chdir(project_root)

try:
    # Import and run main
    from trend_radar.main import main
    
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running this from the project root directory")
    print("💡 Try installing dependencies first: pip install httpx typer[all] rich")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
    