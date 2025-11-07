@echo off
REM Quick start script for Terminal Trading (Windows)

echo ==========================================
echo Terminal Algo Trading System
echo ==========================================
echo.

REM Check if .env exists
if not exist ".env" (
    echo [Warning] .env file not found!
    echo Creating from .env.example...
    copy ..\\.env.example .env
    echo [OK] Created .env - Please edit with your credentials
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import tabulate, colorama, alpaca" >nul 2>&1
if errorlevel 1 (
    echo Installing missing dependencies...
    pip install tabulate colorama alpaca-py
)

echo [OK] All dependencies ready
echo.
echo Starting Terminal Trading...
echo.

REM Run the trading system
python terminal_trading.py

pause
