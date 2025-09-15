@echo off
REM Trend Radar MCP Application Runner Script for Windows
REM This script helps run the application with proper path handling

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    exit /b 1
)

REM Check if required packages are installed
python -c "
try:
    import httpx, typer, rich
    print('✅ Required packages found')
except ImportError as e:
    print('❌ Missing required packages:', e)
    print('💡 Install with: pip install httpx typer[all] rich')
    exit(1)
" || exit /b 1

REM Set Python path to include src directory
set PYTHONPATH=%~dp0src;%PYTHONPATH%

REM Run the application
echo 🎯 Starting Trend Radar MCP Application...
python -m trend_radar.main %*
