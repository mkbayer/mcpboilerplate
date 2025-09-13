#!/usr/bin/env python3
"""
Quick start script for Trend Radar MCP Application
Run this script directly without installation.
"""

import sys
from pathlib import Path

# Add src to path so we can import trend_radar
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run main
from trend_radar.main import main

if __name__ == "__main__":
    main()
    