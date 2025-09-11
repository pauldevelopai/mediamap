#!/bin/bash

# AIMAP Startup Script
echo "🚀 Starting AIMAP Application..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
if [ ! -f ".venv/pyvenv.cfg" ]; then
    echo "📋 Installing requirements..."
    pip install -r requirements.txt
fi

# Check if database exists, if not create it
if [ ! -f "backend/instance/media_analysis.db" ]; then
    echo "🗄️  Creating database..."
    python -c "from backend.create_tables import create_tables; create_tables()"
fi

# Start the application
echo "🌟 Starting Flask application..."
echo "📍 App will be available at: http://localhost:8000"
echo "🔑 Admin login: admin / admin123"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run the Flask app
python -m backend.app










