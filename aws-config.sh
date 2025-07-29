#!/bin/bash

# AWS Configuration for MediaMap Deployment
# This script loads AWS credentials from your existing configuration

# Get AWS account info from existing configuration
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "NOT_CONFIGURED")
REGION=$(aws configure get region || echo "us-east-1")

# Instance Details (if using existing EC2 instance)
INSTANCE_ID="i-09086b9d6aaf71e21"
INSTANCE_IP="54.87.58.143"

# ECS Configuration (if using ECS)
ECR_REPO="mediamap"
CLUSTER_NAME="mediamap-cluster"
SERVICE_NAME="mediamap-service"

echo "✅ AWS configuration loaded:"
echo "   Account ID: $ACCOUNT_ID"
echo "   Region: $REGION"
echo "   Instance: $INSTANCE_ID"
echo "   IP: $INSTANCE_IP"

# Check if credentials are valid
if [ "$ACCOUNT_ID" == "NOT_CONFIGURED" ]; then
    echo "❌ AWS credentials not configured or invalid."
    echo "📝 Please run 'aws configure' to set up your credentials."
    exit 1
fi 