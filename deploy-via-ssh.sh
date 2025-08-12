#!/bin/bash
set -e

# Your existing Lightsail instance details
INSTANCE_IP="13.40.124.51"
INSTANCE_NAME="datasafe"

echo "🚀 Deploying DataSafe via direct SSH..."
echo "📋 Instance: $INSTANCE_NAME"
echo "🌐 IP Address: $INSTANCE_IP"

# Check if we can reach the instance
echo "🔍 Testing connection to instance..."
if ! ping -c 1 $INSTANCE_IP > /dev/null 2>&1; then
    echo "❌ Cannot reach instance at $INSTANCE_IP"
    echo "💡 Please check:"
    echo "   1. Instance is running in Lightsail console"
    echo "   2. Network connectivity"
    exit 1
fi

echo "✅ Instance is reachable!"

# Check for SSH key
echo "🔑 Looking for SSH key..."
KEY_FILES=$(ls *.pem 2>/dev/null || echo "")

if [ -z "$KEY_FILES" ]; then
    echo "❌ No .pem key files found in current directory."
    echo "📁 Please place your Lightsail SSH key file (.pem) in this directory."
    echo "💡 You can download it from the Lightsail console:"
    echo "   1. Go to your Lightsail console"
    echo "   2. Click on your 'datasafe' instance"
    echo "   3. Go to 'Connect' tab"
    echo "   4. Download the SSH key"
    exit 1
fi

# Use the first .pem file found
KEY_FILE=$(echo "$KEY_FILES" | head -n 1)
echo "🔑 Using key file: $KEY_FILE"

# Set proper permissions
chmod 400 $KEY_FILE

# Test SSH connection
echo "🔐 Testing SSH connection..."
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$INSTANCE_IP 'echo "SSH connection successful"' 2>/dev/null; then
    echo "❌ SSH connection failed!"
    echo "💡 Please check:"
    echo "   1. SSH key file is correct"
    echo "   2. Instance is running"
    echo "   3. Security group allows SSH (port 22)"
    exit 1
fi

echo "✅ SSH connection successful!"

# Create deployment script
echo "📝 Creating deployment script..."
cat > remote-deploy.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting DataSafe deployment on Lightsail instance..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    sudo apt install -y docker.io docker-compose
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -a -G docker ubuntu
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
sudo chown ubuntu:ubuntu /opt/datasafe
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
EOF

# Copy deployment script to instance
echo "📤 Copying deployment script to instance..."
scp -i $KEY_FILE -o StrictHostKeyChecking=no remote-deploy.sh ubuntu@$INSTANCE_IP:/tmp/

# Execute deployment script on instance
echo "🚀 Executing deployment on instance..."
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "chmod +x /tmp/remote-deploy.sh && /tmp/remote-deploy.sh"

# Clean up
rm -f remote-deploy.sh

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Summary:"
echo "  - Instance Name: $INSTANCE_NAME"
echo "  - Public IP: $INSTANCE_IP"
echo ""
echo "🌐 Your DataSafe application:"
echo "   http://$INSTANCE_IP"
echo "   http://$INSTANCE_IP:8000 (direct Flask app)"
echo ""
echo "🔍 Management Commands:"
echo "  # SSH into instance"
echo "  ssh -i $KEY_FILE ubuntu@$INSTANCE_IP"
echo ""
echo "  # View logs"
echo "  ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'docker-compose logs -f'"
echo ""
echo "  # Restart application"
echo "  ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'cd /opt/datasafe && docker-compose restart'"
echo ""
echo "⚠️ Note: This deployment replaces the WordPress installation with your DataSafe application." 