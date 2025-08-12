#!/bin/bash

# DataSafe Production Deployment Script
set -e

echo "🚀 Starting DataSafe Production Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p instance
mkdir -p backend/training/models
mkdir -p ssl

# Set environment variables
export FLASK_ENV=production
export FLASK_APP=backend/app.py

# Build and start the application
echo "🔨 Building Docker containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for the application to be ready
echo "⏳ Waiting for application to be ready..."
sleep 30

# Check if the application is running
echo "🔍 Checking application health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Application is healthy!"
    echo "🌐 Your DataSafe application is now running at:"
    echo "   - Local: http://localhost"
    echo "   - Health Check: http://localhost:8000/health"
    echo ""
    echo "📊 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Application health check failed!"
    echo "📋 Checking logs..."
    docker-compose logs
    exit 1
fi

echo "🎉 Deployment completed successfully!" 