#!/bin/bash
set -e

echo "🚀 Starting DataSafe deployment..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    sudo apt install -y docker.io docker-compose
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -a -G docker $USER
    echo "✅ Docker installed!"
else
    echo "✅ Docker already installed"
fi

# Install additional dependencies
echo "📚 Installing additional dependencies..."
sudo apt install -y curl git python3-pip htop

# Stop any existing WordPress services
echo "🛑 Stopping existing WordPress services..."
sudo systemctl stop apache2 2>/dev/null || true
sudo systemctl stop mysql 2>/dev/null || true
sudo systemctl disable apache2 2>/dev/null || true
sudo systemctl disable mysql 2>/dev/null || true

# Create application directory
echo "📁 Setting up application directory..."
sudo mkdir -p /opt/datasafe
sudo chown $USER:$USER /opt/datasafe
cd /opt/datasafe

# Clone or update repository
if [ -d ".git" ]; then
    echo "📥 Updating existing repository..."
    git pull origin main
else
    echo "📥 Cloning repository..."
    git clone https://github.com/pauldevelopai/datasafe.git .
fi

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating environment file..."
    cat > .env << 'ENVEOF'
SECRET_KEY=your-super-secret-production-key-change-this
OPENAI_API_KEY=your-openai-api-key-here
FLASK_ENV=production
ENVEOF
    echo "⚠️ Please edit .env file with your actual values!"
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Build and start application
echo "🔨 Building and starting application..."
docker-compose build --no-cache
docker-compose up -d

# Wait for application to start
echo "⏳ Waiting for application to start..."
sleep 30

# Check health
echo "🏥 Checking application health..."
if curl -f http://localhost:8000/health; then
    echo "✅ Application is healthy!"
else
    echo "❌ Application health check failed!"
    echo "📋 Checking logs..."
    docker-compose logs --tail=20
    exit 1
fi

echo "🎉 Deployment complete!"
echo "🌐 Application should be available at: http://$(curl -s ifconfig.me)"
echo "📊 Check status: docker-compose ps"
echo "📋 View logs: docker-compose logs -f"
