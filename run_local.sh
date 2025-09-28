#!/bin/bash
# 🚀 Local Development Startup Script

echo "🎯 Starting Monte Carlo Prisoner's Dilemma Simulator"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python3 -c "import flask, torch" 2>/dev/null; then
    echo "❌ Dependencies not found! Installing..."
    pip install -r requirements.txt
fi

# Kill any existing processes on port 8000
echo "🧹 Cleaning up port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Start the application
echo "🚀 Starting application on http://localhost:8000"
echo ""
echo "✅ Application is running!"
echo "📱 Open your browser and go to: http://localhost:8000"
echo ""
echo "🎯 To use the app:"
echo "1. Click the 'Start Simulation' button"
echo "2. Watch the real-time progress updates"
echo "3. See live charts and statistics"
echo "4. Analyze the 10,000 experiment results"
echo ""
echo "Press Ctrl+C to stop the server"

# Start the app
python3 app.py
