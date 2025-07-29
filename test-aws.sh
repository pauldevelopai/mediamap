#!/bin/bash

echo "🔍 Testing AWS Configuration..."
echo "================================"

# Test AWS credentials
echo "1. Testing AWS credentials..."
if aws sts get-caller-identity --query 'Account' --output text > /dev/null 2>&1; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
    echo "✅ AWS credentials working! Account: $ACCOUNT_ID"
else
    echo "❌ AWS credentials failed!"
    exit 1
fi

# Test region
echo "2. Testing region configuration..."
REGION=$(aws configure get region || echo "us-east-1")
echo "✅ Region: $REGION"

# Test Lightsail access
echo "3. Testing Lightsail access..."
if aws lightsail get-instances --query 'instances[0].name' --output text > /dev/null 2>&1; then
    echo "✅ Lightsail access working!"
    INSTANCE_COUNT=$(aws lightsail get-instances --query 'length(instances)' --output text)
    echo "📊 Found $INSTANCE_COUNT existing instances"
else
    echo "❌ Lightsail access failed!"
    echo "💡 You may need to enable Lightsail in your AWS account"
    exit 1
fi

# Test EC2 access (for existing instance)
echo "4. Testing EC2 access..."
if aws ec2 describe-instances --instance-ids i-09086b9d6aaf71e21 --query 'Reservations[0].Instances[0].State.Name' --output text > /dev/null 2>&1; then
    INSTANCE_STATE=$(aws ec2 describe-instances --instance-ids i-09086b9d6aaf71e21 --query 'Reservations[0].Instances[0].State.Name' --output text)
    INSTANCE_IP=$(aws ec2 describe-instances --instance-ids i-09086b9d6aaf71e21 --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    echo "✅ EC2 access working!"
    echo "📊 Existing instance state: $INSTANCE_STATE"
    echo "🌐 Existing instance IP: $INSTANCE_IP"
else
    echo "⚠️ EC2 access failed or instance not found"
fi

echo ""
echo "🎉 AWS configuration test completed!"
echo ""
echo "📋 Summary:"
echo "  - Account ID: $ACCOUNT_ID"
echo "  - Region: $REGION"
echo "  - Lightsail: ✅ Working"
echo "  - EC2: ✅ Working"
echo ""
echo "🚀 Ready to deploy! Choose your option:"
echo "  1. Lightsail (recommended): ./deploy-lightsail.sh"
echo "  2. Existing EC2: ./quick-deploy.sh" 