#!/bin/bash
# Script to run the MediaMap Flask app easily from the project root

export FLASK_APP=app.py
export FLASK_ENV=development
cd backend

# Find a free port starting from 5000
PORT=5000
while lsof -i :$PORT >/dev/null 2>&1; do
  PORT=$((PORT+1))
done

echo "Starting MediaMap Flask app on port $PORT..."
flask run --port=$PORT 