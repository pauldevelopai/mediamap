#!/bin/bash
set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPO="datasafe"
CLUSTER_NAME="datasafe-cluster"
SERVICE_NAME="datasafe-service"

echo "🚀 Starting DataSafe ECS deployment..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "📋 Using AWS Account: $ACCOUNT_ID"

# Check if ECR repository exists, create if not
if ! aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION &> /dev/null; then
    echo "📦 Creating ECR repository..."
    aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t $ECR_REPO .

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag and push image
echo "📤 Tagging and pushing image..."
docker tag $ECR_REPO:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# Check if ECS cluster exists, create if not
if ! aws ecs describe-clusters --clusters $CLUSTER_NAME --region $AWS_REGION --query 'clusters[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
    echo "🏗️ Creating ECS cluster..."
    aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $AWS_REGION
fi

# Check if ECS service exists
if aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION --query 'services[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
    echo "🔄 Updating existing ECS service..."
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --force-new-deployment \
        --region $AWS_REGION
else
    echo "⚠️ ECS service '$SERVICE_NAME' not found. Please create it first using the AWS console or CLI."
    echo "📖 See aws-deployment-guide.md for detailed instructions."
    exit 1
fi

echo "✅ Deployment initiated! Service is updating..."
echo "📊 Monitor deployment: aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION" 