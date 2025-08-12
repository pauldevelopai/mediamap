#!/bin/bash
set -e

echo "🚀 Starting DataSafe EC2 deployment..."

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found. Please run this script from the project root."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found. Creating template..."
    cat > .env << EOF
SECRET_KEY=your-super-secret-production-key-change-this
OPENAI_API_KEY=your-openai-api-key-here
FLASK_ENV=production
EOF
    echo "📝 Please edit .env file with your actual values before continuing."
    exit 1
fi

# Pull latest code
echo "📥 Pulling latest code..."
git pull origin main

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build new image
echo "🔨 Building new Docker image..."
docker-compose build --no-cache

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check health
echo "🏥 Checking application health..."
if curl -f http://localhost:8000/health; then
    echo "✅ Application is healthy!"
else
    echo "❌ Application health check failed!"
    echo "📋 Checking logs..."
    docker-compose logs
    exit 1
fi

echo "🎉 Deployment complete!"
echo "🌐 Application should be available at: http://localhost"
echo "📊 Check status: docker-compose ps"
echo "📋 View logs: docker-compose logs -f" 