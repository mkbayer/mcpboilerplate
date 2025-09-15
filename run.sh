#!/bin/bash

# Trend Radar MCP Application Runner Script
# This script helps run the application with proper path handling

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
python3 -c "
try:
    import httpx, typer, rich
    print('✅ Required packages found')
except ImportError as e:
    print('❌ Missing required packages:', e)
    print('💡 Install with: pip install httpx typer[all] rich')
    exit(1)
" || exit 1

# Set Python path to include src directory
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Run the application
echo "🎯 Starting Trend Radar MCP Application..."
python3 -m trend_radar.main "$@"
