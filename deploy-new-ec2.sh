#!/bin/bash
set -e

# EC2 Configuration
INSTANCE_TYPE="t3.micro"  # 1GB RAM, 2 vCPU - Free tier eligible
AMI_ID="ami-0c02fb55956c7d316"  # Ubuntu 22.04 LTS
KEY_PAIR_NAME="mediamap-key"
SECURITY_GROUP_NAME="mediamap-sg"

echo "🚀 Deploying MediaMap to new EC2 instance..."
echo "📋 Instance Type: $INSTANCE_TYPE"
echo "💰 Cost: ~$8-15/month (or free with AWS Free Tier)"

# Verify AWS credentials
echo "🔐 Verifying AWS credentials..."
if ! aws sts get-caller-identity --query 'Account' --output text > /dev/null 2>&1; then
    echo "❌ AWS credentials not configured or invalid."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
REGION=$(aws configure get region || echo "us-east-1")
echo "✅ AWS credentials verified! Account: $ACCOUNT_ID"

# Get default VPC
echo "🌐 Getting VPC information..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0].SubnetId' --output text)

echo "✅ VPC: $VPC_ID, Subnet: $SUBNET_ID"

# Create key pair
echo "🔑 Creating key pair..."
if ! aws ec2 describe-key-pairs --key-names $KEY_PAIR_NAME --query 'KeyPairs[0].KeyName' --output text > /dev/null 2>&1; then
    aws ec2 create-key-pair --key-name $KEY_PAIR_NAME --query 'KeyMaterial' --output text > $KEY_PAIR_NAME.pem
    chmod 400 $KEY_PAIR_NAME.pem
    echo "✅ Key pair created: $KEY_PAIR_NAME.pem"
else
    echo "✅ Key pair already exists"
fi

# Create security group
echo "🛡️ Creating security group..."
if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --query 'SecurityGroups[0].GroupName' --output text > /dev/null 2>&1; then
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for MediaMap application" \
        --vpc-id $VPC_ID \
        --query 'GroupId' --output text)
    
    # Allow SSH
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0
    
    # Allow HTTP
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0
    
    # Allow HTTPS
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0
    
    # Allow application port
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0
    
    echo "✅ Security group created: $SECURITY_GROUP_ID"
else
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --query 'SecurityGroups[0].GroupId' --output text)
    echo "✅ Security group already exists: $SECURITY_GROUP_ID"
fi

# Create user data script
echo "📝 Creating user data script..."
USER_DATA=$(cat << 'EOF'
#!/bin/bash
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
mkdir -p /opt/mediamap
chown ubuntu:ubuntu /opt/mediamap

# Clone MediaMap repository
cd /opt/mediamap
git clone https://github.com/pauldevelopai/mediamap.git .

# Create environment file
cat > .env << 'ENVEOF'
SECRET_KEY=your-super-secret-production-key-change-this
OPENAI_API_KEY=your-openai-api-key-here
FLASK_ENV=production
ENVEOF

# Build and start application
docker-compose build --no-cache
docker-compose up -d

# Wait for application to start
sleep 30

# Check health
curl -f http://localhost:8000/health || echo "Health check failed"
EOF
)

# Launch instance
echo "🚀 Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_PAIR_NAME \
    --security-group-ids $SECURITY_GROUP_ID \
    --subnet-id $SUBNET_ID \
    --user-data "$USER_DATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mediamap-server}]' \
    --query 'Instances[0].InstanceId' --output text)

echo "✅ Instance launched: $INSTANCE_ID"

# Wait for instance to be running
echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
INSTANCE_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "✅ Instance is running!"
echo "🌐 Public IP: $INSTANCE_IP"

# Wait for user data script to complete
echo "⏳ Waiting for deployment to complete..."
sleep 120

# Test the application
echo "🏥 Testing application..."
if curl -f http://$INSTANCE_IP:8000/health 2>/dev/null; then
    echo "✅ Application is healthy!"
else
    echo "⚠️ Application health check failed. Checking logs..."
    ssh -i $KEY_PAIR_NAME.pem -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP 'docker-compose logs --tail=20'
fi

echo ""
echo "🎉 EC2 deployment completed!"
echo ""
echo "📋 Summary:"
echo "  - Instance ID: $INSTANCE_ID"
echo "  - Public IP: $INSTANCE_IP"
echo "  - Instance Type: $INSTANCE_TYPE"
echo "  - Cost: ~$8-15/month (or free with AWS Free Tier)"
echo ""
echo "🌐 Your MediaMap application:"
echo "   http://$INSTANCE_IP"
echo "   http://$INSTANCE_IP:8000 (direct Flask app)"
echo ""
echo "🔍 Management Commands:"
echo "  # Check status"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID"
echo ""
echo "  # SSH into instance"
echo "  ssh -i $KEY_PAIR_NAME.pem ubuntu@$INSTANCE_IP"
echo ""
echo "  # View logs"
echo "  ssh -i $KEY_PAIR_NAME.pem ubuntu@$INSTANCE_IP 'docker-compose logs -f'"
echo ""
echo "  # Restart application"
echo "  ssh -i $KEY_PAIR_NAME.pem ubuntu@$INSTANCE_IP 'cd /opt/mediamap && docker-compose restart'"
echo ""
echo "💰 Cost Information:"
echo "  - t3.micro: Free tier eligible (750 hours/month)"
echo "  - After free tier: ~$8-15/month"
echo "  - Data transfer: First 1GB free, then $0.09/GB" 