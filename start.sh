#!/bin/bash

# AIMAP Startup Script with Auto Chrome Opening
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

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "python.*backend.app" 2>/dev/null || true
sleep 2

# Function to find available port
find_available_port() {
    for port in {3000..8100}; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo $port
            return
        fi
    done
    echo "3000"  # fallback
}

# Find available port
PORT=$(find_available_port)
echo "🌟 Starting Flask application on port $PORT..."

# Start the Flask app in background
python -m backend.app > app.log 2>&1 &
APP_PID=$!

# Function to wait for app to be ready
wait_for_app() {
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for application to start..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$PORT" >/dev/null 2>&1; then
            echo "✅ Application is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ Application failed to start within 60 seconds"
    return 1
}

# Wait for app to be ready
if wait_for_app; then
    echo ""
    echo "🎉 AIMAP Application Started Successfully!"
    echo "📍 App URL: http://localhost:$PORT"
    echo "🔑 Admin login: admin / admin123"
    echo "🎯 Opening MediaMap admin page in Chrome..."
    echo ""
    
    # Open Chrome to admin MediaMap page
    ADMIN_URL="http://localhost:$PORT/admin/map"
    
    # Try different Chrome commands for different systems
    if command -v google-chrome >/dev/null 2>&1; then
        google-chrome --new-window "$ADMIN_URL" >/dev/null 2>&1 &
    elif command -v google-chrome-stable >/dev/null 2>&1; then
        google-chrome-stable --new-window "$ADMIN_URL" >/dev/null 2>&1 &
    elif command -v chromium-browser >/dev/null 2>&1; then
        chromium-browser --new-window "$ADMIN_URL" >/dev/null 2>&1 &
    elif command -v open >/dev/null 2>&1; then
        # macOS
        open -a "Google Chrome" "$ADMIN_URL" >/dev/null 2>&1 &
    else
        echo "⚠️  Chrome not found. Please manually open: $ADMIN_URL"
    fi
    
    echo "🌐 MediaMap Admin: $ADMIN_URL"
    echo "📊 Chat Management: http://localhost:$PORT/admin/chat-management"
    echo "🤖 AI Agents: http://localhost:$PORT/admin/agents"
    echo ""
    echo "Press Ctrl+C to stop the application"
    echo ""
    
    # Wait for the background process
    wait $APP_PID
else
    echo "❌ Failed to start application"
    kill $APP_PID 2>/dev/null || true
    exit 1
fi

















