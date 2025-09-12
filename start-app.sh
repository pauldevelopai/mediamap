#!/bin/bash

# MediaMap Flask App Startup Script
# This script automatically handles port conflicts and starts the app

echo "🎯 MediaMap Flask App Startup"
echo "=============================="

# Navigate to backend directory
cd "$(dirname "$0")/backend" || {
    echo "❌ Error: Could not find backend directory"
    exit 1
}

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in backend directory"
    exit 1
fi

# Kill any existing Flask processes
echo "🧹 Cleaning up existing processes..."
pkill -f "python app.py" 2>/dev/null || true
sleep 2

# Start the Flask app
echo "🚀 Starting Flask app..."
python app.py

echo "👋 Flask app has stopped"
