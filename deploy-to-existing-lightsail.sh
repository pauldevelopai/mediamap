#!/bin/bash
set -e

echo "🚀 Deploying MediaMap to Existing Lightsail Instance"
echo "=================================================="
echo ""

# Configuration
INSTANCE_NAME="MEDIAMAPUPGRADE"
REGION="eu-west-2"
APP_DIR="/opt/mediamap"

echo "📋 Instance: $INSTANCE_NAME"
echo "🌍 Region: $REGION"
echo ""

# Get instance details from console
echo "🔍 Please get the following information from your Lightsail console:"
echo "   1. Public IP address"
echo "   2. SSH key name"
echo ""
echo "🌐 Go to: https://lightsail.aws.amazon.com/ls/webapp/$REGION/instances/$INSTANCE_NAME/connect"
echo ""

# Prompt for instance details
read -p "Enter the Public IP address: " INSTANCE_IP
read -p "Enter the SSH key name (e.g., LightsailDefaultKey-eu-west-2): " KEY_NAME

# Check if SSH key exists locally
KEY_FILE="$KEY_NAME.pem"
if [ ! -f "$KEY_FILE" ]; then
    echo "❌ SSH key file not found: $KEY_FILE"
    echo "📥 Please download the SSH key from Lightsail console:"
    echo "   1. Go to your instance"
    echo "   2. Click 'Connect using SSH'"
    echo "   3. Download the key file"
    echo "   4. Place it in this directory as: $KEY_FILE"
    echo ""
    read -p "Press Enter when you have the key file ready..."
fi

# Set proper permissions
chmod 400 $KEY_FILE

echo "🔑 Using SSH key: $KEY_FILE"
echo "🌐 Connecting to: ubuntu@$INSTANCE_IP"
echo ""

# Test SSH connection
echo "🔍 Testing SSH connection..."
if ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$INSTANCE_IP 'echo "SSH connection successful!"'; then
    echo "✅ SSH connection successful!"
else
    echo "❌ SSH connection failed. Please check:"
    echo "   - IP address is correct"
    echo "   - SSH key file exists and has correct permissions"
    echo "   - Instance is running"
    exit 1
fi

echo ""
echo "🚀 Starting deployment..."
echo "⏳ This will take 5-10 minutes..."
echo ""

# Create deployment script
cat > deploy_script.sh << 'EOF'
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
EOF

# Copy and execute deployment script
echo "📤 Uploading deployment script..."
scp -i $KEY_FILE deploy_script.sh ubuntu@$INSTANCE_IP:/tmp/

echo "🚀 Executing deployment on server..."
ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'chmod +x /tmp/deploy_script.sh && /tmp/deploy_script.sh'

# Clean up
rm deploy_script.sh

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Summary:"
echo "  - Instance: $INSTANCE_NAME"
echo "  - Public IP: $INSTANCE_IP"
echo "  - Region: $REGION"
echo ""
echo "🌐 Your MediaMap application:"
echo "   http://$INSTANCE_IP:8000"
echo ""
echo "🔍 Management Commands:"
echo "  # Check status"
echo "  ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'docker-compose ps'"
echo ""
echo "  # View logs"
echo "  ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'docker-compose logs -f'"
echo ""
echo "  # Restart application"
echo "  ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'cd /opt/mediamap && docker-compose restart'"
echo ""
echo "  # Update application"
echo "  ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'cd /opt/mediamap && git pull && docker-compose build --no-cache && docker-compose up -d'"
echo ""
echo "🔧 Next Steps:"
echo "  1. Visit http://$INSTANCE_IP:8000 to access your application"
echo "  2. Edit /opt/mediamap/.env to add your OpenAI API key"
echo "  3. Restart the application: docker-compose restart"
echo ""
echo "💰 Cost: Your Lightsail instance is already running and billing"