#!/bin/bash
set -e

echo "🔧 Updating system..."
sudo apt update && sudo apt upgrade -y

echo "🐳 Installing Docker..."
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu

echo "📦 Installing additional tools..."
sudo apt install -y curl git python3-pip htop

echo "📁 Creating application directory..."
sudo mkdir -p /opt/mediamap
sudo chown ubuntu:ubuntu /opt/mediamap

echo "📥 Cloning MediaMap repository..."
cd /opt/mediamap
git clone https://github.com/pauldevelopai/mediamap.git .

echo "🔧 Creating environment file..."
cat > .env << 'ENVEOF'
SECRET_KEY=your-super-secret-production-key-change-this
OPENAI_API_KEY=your-openai-api-key-here
FLASK_ENV=production
DATABASE_URL=sqlite:///./instance/media_analysis.db
HUGGINGFACE_HUB_TOKEN=
ENVEOF

echo "🐳 Building and starting application..."
docker-compose build --no-cache
docker-compose up -d

echo "⏳ Waiting for application to start..."
sleep 30

echo "🏥 Testing application health..."
if curl -f http://localhost:8000/health; then
    echo "✅ Application is healthy!"
else
    echo "⚠️ Health check failed, but application may still be starting..."
fi

echo "📊 Checking running containers..."
docker-compose ps

echo "🎉 Deployment completed!"
echo "🌐 Your MediaMap application should be available at:"
echo "   http://$(curl -s ifconfig.me):8000"
echo ""
echo "🔍 To check logs:"
echo "   docker-compose logs -f"
echo ""
echo "🔧 To restart:"
echo "   docker-compose restart"
