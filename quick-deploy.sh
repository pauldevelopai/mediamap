#!/bin/bash
set -e

echo "🚀 Quick Deploy to Existing AWS Instance"
echo "========================================"

# Load AWS configuration
source ./aws-config.sh

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not installed. Please install it first:"
    echo "   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Verify AWS credentials
echo "🔐 Verifying AWS credentials..."
aws sts get-caller-identity

# Check instance status
echo "🔍 Checking instance status..."
INSTANCE_STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text)

if [ "$INSTANCE_STATE" != "running" ]; then
    echo "❌ Instance is not running. Current state: $INSTANCE_STATE"
    echo "🚀 Starting instance..."
    aws ec2 start-instances --instance-ids $INSTANCE_ID
    echo "⏳ Waiting for instance to start..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    echo "✅ Instance is now running!"
else
    echo "✅ Instance is already running!"
fi

# Get security group ID
SECURITY_GROUP_ID=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)

# Ensure port 80 and 8000 are open
echo "🔓 Ensuring ports are open..."
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 2>/dev/null || echo "Port 80 already open"

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0 2>/dev/null || echo "Port 8000 already open"

# Check for SSH key
echo "🔑 Looking for SSH key files..."
KEY_FILES=$(ls *.pem 2>/dev/null || echo "")

if [ -z "$KEY_FILES" ]; then
    echo "❌ No .pem key files found in current directory."
    echo "📁 Please place your SSH key file (.pem) in this directory."
    echo "🔍 Available key pairs in AWS:"
    aws ec2 describe-key-pairs --query 'KeyPairs[*].KeyName' --output table
    exit 1
fi

# Use the first .pem file found
KEY_FILE=$(echo "$KEY_FILES" | head -n 1)
echo "🔑 Using key file: $KEY_FILE"

# Set proper permissions
chmod 400 $KEY_FILE

echo "🚀 Starting deployment..."
echo "📤 This will install DataSafe on your instance at: $INSTANCE_IP"
echo "⏳ This may take 5-10 minutes..."

# Execute the deployment
./deploy-to-existing-instance.sh

echo ""
echo "🎉 Deployment completed!"
echo "🌐 Your DataSafe application should be available at:"
echo "   http://$INSTANCE_IP"
echo ""
echo "🔍 To check status:"
echo "   ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'docker-compose ps'"
echo ""
echo "📋 To view logs:"
echo "   ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'docker-compose logs -f'" 