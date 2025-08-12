#!/bin/bash
set -e

# Lightsail Configuration
LIGHTSAIL_INSTANCE_NAME="datasafe-server"
LIGHTSAIL_BLUEPRINT="ubuntu_22_04"
LIGHTSAIL_BUNDLE="nano_2_0"  # 512MB RAM, 1 vCPU, 20GB SSD - $3.50/month
# Alternative bundles: micro_2_0 (1GB RAM, 1 vCPU, 40GB SSD - $7/month)

echo "🚀 Deploying DataSafe to AWS Lightsail..."
echo "📋 Instance: $LIGHTSAIL_INSTANCE_NAME"
echo "💰 Bundle: $LIGHTSAIL_BUNDLE (~$3.50/month)"

# Verify AWS credentials are configured
echo "🔐 Verifying AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured or invalid."
    echo "📝 Please run 'aws configure' to set up your credentials."
    echo "   Or ensure your credentials are properly configured in ~/.aws/credentials"
    exit 1
fi

# Get AWS account info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region || echo "us-east-1")
echo "✅ AWS credentials verified!"
echo "📋 Account ID: $ACCOUNT_ID"
echo "🌍 Region: $REGION"

# Check if instance already exists
echo "🔍 Checking for existing Lightsail instance..."
EXISTING_INSTANCE=$(aws lightsail get-instances --instance-names $LIGHTSAIL_INSTANCE_NAME --query 'instances[0].name' --output text 2>/dev/null || echo "")

if [ "$EXISTING_INSTANCE" == "$LIGHTSAIL_INSTANCE_NAME" ]; then
    echo "✅ Instance already exists!"
    
    # Get instance details
    INSTANCE_STATE=$(aws lightsail get-instances --instance-names $LIGHTSAIL_INSTANCE_NAME --query 'instances[0].state.name' --output text)
    INSTANCE_IP=$(aws lightsail get-instances --instance-names $LIGHTSAIL_INSTANCE_NAME --query 'instances[0].publicIpAddress' --output text)
    
    echo "📊 Instance State: $INSTANCE_STATE"
    echo "🌐 Public IP: $INSTANCE_IP"
    
    if [ "$INSTANCE_STATE" != "running" ]; then
        echo "🚀 Starting instance..."
        aws lightsail start-instance --instance-name $LIGHTSAIL_INSTANCE_NAME
        echo "⏳ Waiting for instance to start..."
        aws lightsail wait instance-running --instance-name $LIGHTSAIL_INSTANCE_NAME
        echo "✅ Instance is now running!"
    fi
else
    echo "🏗️ Creating new Lightsail instance..."
    
    # Create instance
    aws lightsail create-instances \
        --instance-names $LIGHTSAIL_INSTANCE_NAME \
        --availability-zone ${REGION}a \
        --blueprint-id $LIGHTSAIL_BLUEPRINT \
        --bundle-id $LIGHTSAIL_BUNDLE \
        --user-data '#!/bin/bash
# Update system
apt update && apt upgrade -y

# Install Docker
apt install -y docker.io docker-compose
systemctl start docker
systemctl enable docker
usermod -a -G docker ubuntu

# Install additional tools
apt install -y curl git python3-pip htop

# Create application directory
mkdir -p /opt/datasafe
chown ubuntu:ubuntu /opt/datasafe

# Clone DataSafe repository
cd /opt/datasafe
git clone https://github.com/pauldevelopai/datasafe.git .

# Create environment file
cat > .env << EOF
SECRET_KEY=your-super-secret-production-key-change-this
OPENAI_API_KEY=your-openai-api-key-here
FLASK_ENV=production
EOF

# Build and start application
docker-compose build --no-cache
docker-compose up -d

# Wait for application to start
sleep 30

# Check health
curl -f http://localhost:8000/health || echo "Health check failed"
'
    
    echo "⏳ Waiting for instance to be ready..."
    aws lightsail wait instance-running --instance-name $LIGHTSAIL_INSTANCE_NAME
    
    # Get instance IP
    INSTANCE_IP=$(aws lightsail get-instances --instance-names $LIGHTSAIL_INSTANCE_NAME --query 'instances[0].publicIpAddress' --output text)
    
    echo "✅ Instance created successfully!"
    echo "🌐 Public IP: $INSTANCE_IP"
fi

# Open ports (HTTP and HTTPS)
echo "🔓 Opening ports..."
aws lightsail open-instance-public-ports \
    --instance-name $LIGHTSAIL_INSTANCE_NAME \
    --port-info fromPort=80,toPort=80,protocol=tcp 2>/dev/null || echo "Port 80 already open"

aws lightsail open-instance-public-ports \
    --instance-name $LIGHTSAIL_INSTANCE_NAME \
    --port-info fromPort=443,toPort=443,protocol=tcp 2>/dev/null || echo "Port 443 already open"

aws lightsail open-instance-public-ports \
    --instance-name $LIGHTSAIL_INSTANCE_NAME \
    --port-info fromPort=8000,toPort=8000,protocol=tcp 2>/dev/null || echo "Port 8000 already open"

# Get the SSH key
echo "🔑 Getting SSH key..."
KEY_PAIR_NAME=$(aws lightsail get-instances --instance-names $LIGHTSAIL_INSTANCE_NAME --query 'instances[0].sshKeyName' --output text)

# Download SSH key if not exists
if [ ! -f "$KEY_PAIR_NAME.pem" ]; then
    echo "📥 Downloading SSH key..."
    aws lightsail download-default-key-pair --output text > $KEY_PAIR_NAME.pem
    chmod 400 $KEY_PAIR_NAME.pem
fi

# Wait a bit for the user data script to complete
echo "⏳ Waiting for deployment to complete..."
sleep 60

# Test the application
echo "🏥 Testing application..."
if curl -f http://$INSTANCE_IP:8000/health 2>/dev/null; then
    echo "✅ Application is healthy!"
else
    echo "⚠️ Application health check failed. Checking logs..."
    ssh -i $KEY_PAIR_NAME.pem -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP 'docker-compose logs --tail=20'
fi

echo ""
echo "🎉 Lightsail deployment completed!"
echo ""
echo "📋 Summary:"
echo "  - Instance Name: $LIGHTSAIL_INSTANCE_NAME"
echo "  - Public IP: $INSTANCE_IP"
echo "  - Bundle: $LIGHTSAIL_BUNDLE"
echo "  - Cost: ~$3.50/month"
echo ""
echo "🌐 Your DataSafe application:"
echo "   http://$INSTANCE_IP"
echo "   http://$INSTANCE_IP:8000 (direct Flask app)"
echo ""
echo "🔍 Management Commands:"
echo "  # Check status"
echo "  aws lightsail get-instances --instance-names $LIGHTSAIL_INSTANCE_NAME"
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
echo "💰 Cost Information:"
echo "  - Lightsail $LIGHTSAIL_BUNDLE: ~$3.50/month"
echo "  - Data transfer: First 1TB free, then $0.09/GB"
echo "  - Total estimated cost: $3.50-5.00/month" 