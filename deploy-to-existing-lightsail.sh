#!/bin/bash
set -e

# Your existing Lightsail instance details
LIGHTSAIL_INSTANCE_NAME="datasafe"
INSTANCE_IP="13.40.124.51"
REGION="eu-west-2"

echo "🚀 Deploying DataSafe to your existing Lightsail instance..."
echo "📋 Instance: $LIGHTSAIL_INSTANCE_NAME"
echo "🌐 IP Address: $INSTANCE_IP"
echo "🌍 Region: $REGION"

# Verify AWS credentials
echo "🔐 Verifying AWS credentials..."
if ! aws sts get-caller-identity --query 'Account' --output text > /dev/null 2>&1; then
    echo "❌ AWS credentials not configured or invalid."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
echo "✅ AWS credentials verified! Account: $ACCOUNT_ID"

# Check Lightsail permissions
echo "🔍 Checking Lightsail permissions..."
if ! aws lightsail get-instances --region $REGION --query 'instances[0].name' --output text > /dev/null 2>&1; then
    echo "❌ No Lightsail permissions or no instances found in region $REGION"
    echo "💡 You may need to:"
    echo "   1. Enable Lightsail in your AWS account"
    echo "   2. Add Lightsail permissions to your IAM user"
    echo "   3. Or use the AWS Console to manage your instance"
    echo ""
    echo "🔧 Alternative: Deploy directly via SSH"
    echo "   Since we know your instance IP, we can deploy directly:"
    echo "   ./deploy-via-ssh.sh"
    exit 1
fi

# Check if instance exists and is running
echo "🔍 Checking instance status..."
INSTANCE_STATE=$(aws lightsail get-instances --instance-names $LIGHTSAIL_INSTANCE_NAME --region $REGION --query 'instances[0].state.name' --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$INSTANCE_STATE" == "NOT_FOUND" ]; then
    echo "❌ Instance '$LIGHTSAIL_INSTANCE_NAME' not found in region $REGION"
    echo "💡 Available instances:"
    aws lightsail get-instances --region $REGION --query 'instances[*].name' --output text 2>/dev/null || echo "None found"
    exit 1
fi

echo "📊 Instance State: $INSTANCE_STATE"

if [ "$INSTANCE_STATE" != "running" ]; then
    echo "🚀 Starting instance..."
    aws lightsail start-instance --instance-name $LIGHTSAIL_INSTANCE_NAME --region $REGION
    echo "⏳ Waiting for instance to start..."
    aws lightsail wait instance-running --instance-name $LIGHTSAIL_INSTANCE_NAME --region $REGION
    echo "✅ Instance is now running!"
fi

# Get SSH key
echo "🔑 Getting SSH key..."
KEY_PAIR_NAME=$(aws lightsail get-instances --instance-names $LIGHTSAIL_INSTANCE_NAME --region $REGION --query 'instances[0].sshKeyName' --output text)

# Download SSH key if not exists
if [ ! -f "$KEY_PAIR_NAME.pem" ]; then
    echo "📥 Downloading SSH key..."
    aws lightsail download-default-key-pair --region $REGION --output text > $KEY_PAIR_NAME.pem
    chmod 400 $KEY_PAIR_NAME.pem
    echo "✅ SSH key downloaded: $KEY_PAIR_NAME.pem"
fi

# Ensure ports are open
echo "🔓 Ensuring ports are open..."
aws lightsail open-instance-public-ports \
    --instance-name $LIGHTSAIL_INSTANCE_NAME \
    --port-info fromPort=80,toPort=80,protocol=tcp \
    --region $REGION 2>/dev/null || echo "Port 80 already open"

aws lightsail open-instance-public-ports \
    --instance-name $LIGHTSAIL_INSTANCE_NAME \
    --port-info fromPort=443,toPort=443,protocol=tcp \
    --region $REGION 2>/dev/null || echo "Port 443 already open"

aws lightsail open-instance-public-ports \
    --instance-name $LIGHTSAIL_INSTANCE_NAME \
    --port-info fromPort=8000,toPort=8000,protocol=tcp \
    --region $REGION 2>/dev/null || echo "Port 8000 already open"

# Create deployment script for remote execution
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
scp -i $KEY_PAIR_NAME.pem -o StrictHostKeyChecking=no remote-deploy.sh ubuntu@$INSTANCE_IP:/tmp/

# Execute deployment script on instance
echo "🚀 Executing deployment on instance..."
ssh -i $KEY_PAIR_NAME.pem -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "chmod +x /tmp/remote-deploy.sh && /tmp/remote-deploy.sh"

# Clean up
rm -f remote-deploy.sh

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Summary:"
echo "  - Instance Name: $LIGHTSAIL_INSTANCE_NAME"
echo "  - Public IP: $INSTANCE_IP"
echo "  - Region: $REGION"
echo ""
echo "🌐 Your DataSafe application:"
echo "   http://$INSTANCE_IP"
echo "   http://$INSTANCE_IP:8000 (direct Flask app)"
echo ""
echo "🔍 Management Commands:"
echo "  # Check status"
echo "  aws lightsail get-instances --instance-names $LIGHTSAIL_INSTANCE_NAME --region $REGION"
echo ""
echo "  # SSH into instance"
echo "  ssh -i $KEY_PAIR_NAME.pem ubuntu@$INSTANCE_IP"
echo ""
echo "  # View logs"
echo "  ssh -i $KEY_PAIR_NAME.pem ubuntu@$INSTANCE_IP 'docker-compose logs -f'"
echo ""
echo "  # Restart application"
echo "  ssh -i $KEY_PAIR_NAME.pem ubuntu@$INSTANCE_IP 'cd /opt/datasafe && docker-compose restart'"
echo ""
echo "⚠️ Note: This deployment replaces the WordPress installation with your DataSafe application." 