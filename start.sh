#!/bin/bash
# 🚀 Monte Carlo Prisoner's Dilemma Startup Script

echo "🎯 Starting Monte Carlo Prisoner's Dilemma Simulator"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask not found! Installing dependencies..."
    pip install -r requirements.txt
fi

# Start Flask backend
echo "🚀 Starting Flask backend on http://127.0.0.1:5000"
cd backend
python app.py &
BACKEND_PID=$!

# Go back to root directory
cd ..

# Start frontend server
echo "🌐 Starting frontend server on http://127.0.0.1:8000"
python -m http.server 8000 &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers are running!"
echo "📱 Frontend: http://127.0.0.1:8000"
echo "🔧 Backend API: http://127.0.0.1:5000"
echo ""
echo "🎯 To run your parameter sweep experiment:"
echo "1. Open http://127.0.0.1:8000 in your browser"
echo "2. Scroll down to 'Parameter Sweep Experiment'"
echo "3. Click 'Run Parameter Sweep'"
echo "4. Watch the 10,000 experiments run!"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
