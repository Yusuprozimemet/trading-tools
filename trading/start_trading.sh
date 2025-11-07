#!/bin/bash
# Quick start script for Terminal Trading

echo "=========================================="
echo "Terminal Algo Trading System"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠ Warning: .env file not found!"
    echo "Creating from .env.example..."
    cp ../.env.example .env
    echo "✓ Created .env - Please edit with your credentials"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import tabulate, colorama, alpaca" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing missing dependencies..."
    pip install tabulate colorama alpaca-py
fi

echo "✓ All dependencies ready"
echo ""
echo "Starting Terminal Trading..."
echo ""

# Run the trading system
python terminal_trading.py
