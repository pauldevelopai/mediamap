#!/bin/bash
set -e

echo "🚀 Starting DataSafe deployment on upgraded instance..."
echo "📦 Updating system packages..."

# Update system
sudo apt update && sudo apt upgrade -y

echo "🐳 Installing Docker..."
# Install Docker
sudo apt install -y docker.io docker-compose curl git htop

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

echo "📁 Setting up application directory..."
# Create app directory
sudo mkdir -p /opt/datasafe
sudo chown ubuntu:ubuntu /opt/datasafe
cd /opt/datasafe

echo "📥 Cloning repository..."
# Clone the repository
git clone https://github.com/pauldevelopai/datasafe.git . || echo "Repository already exists, updating..." && git pull

echo "📝 Creating environment file..."
# Create .env file
cat > .env << 'ENVEOF'
FLASK_ENV=production
FLASK_APP=backend/app.py
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///instance/datasafe.db
ENVEOF

echo "🔨 Installing Python dependencies..."
# Install Python dependencies
python3 -m pip install --user -r requirements.txt

echo "🚀 Starting application..."
# Start the application
nohup python3 backend/app.py --host=0.0.0.0 --port=8000 > app.log 2>&1 &

echo "✅ DataSafe deployment complete!"
echo "🌐 Access your app at: http://$(curl -s ifconfig.me):8000"
echo "📋 Check logs with: tail -f /opt/datasafe/app.log"
