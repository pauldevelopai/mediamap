#!/bin/bash
set -e

# Configuration
AWS_REGION="us-east-1"
CLUSTER_NAME="datasafe-cluster"
SERVICE_NAME="datasafe-service"
SECURITY_GROUP_NAME="datasafe-sg"

echo "🔧 Setting up AWS resources for DataSafe..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "📋 Using AWS Account: $ACCOUNT_ID"

# Get default VPC ID
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text --region $AWS_REGION)
echo "🌐 Using VPC: $VPC_ID"

# Get default subnets
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[*].SubnetId' --output text --region $AWS_REGION)
SUBNET_ARRAY=($SUBNET_IDS)
echo "🔗 Found subnets: ${SUBNET_ARRAY[0]} ${SUBNET_ARRAY[1]}"

# Create security group
echo "🛡️ Creating security group..."
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name $SECURITY_GROUP_NAME \
    --description "Security group for DataSafe application" \
    --vpc-id $VPC_ID \
    --region $AWS_REGION \
    --query 'GroupId' --output text)

echo "✅ Security group created: $SECURITY_GROUP_ID"

# Allow HTTP/HTTPS traffic
echo "🌐 Configuring HTTP/HTTPS access..."
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION

# Allow SSH access (get current IP)
CURRENT_IP=$(curl -s ifconfig.me)
echo "🔑 Allowing SSH access from: $CURRENT_IP"
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 22 \
    --cidr $CURRENT_IP/32 \
    --region $AWS_REGION

# Create ECS task execution role
echo "👤 Creating ECS task execution role..."
aws iam create-role \
    --role-name ecsTaskExecutionRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }' 2>/dev/null || echo "Role already exists"

# Attach policies
aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Create ECS cluster
echo "🏗️ Creating ECS cluster..."
aws ecs create-cluster \
    --cluster-name $CLUSTER_NAME \
    --region $AWS_REGION

# Create ECR repository
echo "📦 Creating ECR repository..."
aws ecr create-repository \
    --repository-name datasafe \
    --region $AWS_REGION

# Create CloudWatch log group
echo "📊 Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name /ecs/datasafe \
    --region $AWS_REGION 2>/dev/null || echo "Log group already exists"

# Create task definition
echo "📋 Creating task definition..."
cat > task-definition.json << EOF
{
  "family": "datasafe",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "datasafe",
      "image": "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/datasafe:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FLASK_ENV",
          "value": "production"
        },
        {
          "name": "FLASK_APP",
          "value": "backend/app.py"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/datasafe",
          "awslogs-region": "$AWS_REGION",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region $AWS_REGION

# Create Application Load Balancer
echo "⚖️ Creating Application Load Balancer..."
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name datasafe-alb \
    --subnets ${SUBNET_ARRAY[0]} ${SUBNET_ARRAY[1]} \
    --security-groups $SECURITY_GROUP_ID \
    --region $AWS_REGION \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text)

echo "✅ Load balancer created: $ALB_ARN"

# Create target group
echo "🎯 Creating target group..."
TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name datasafe-tg \
    --protocol HTTP \
    --port 8000 \
    --vpc-id $VPC_ID \
    --target-type ip \
    --region $AWS_REGION \
    --query 'TargetGroups[0].TargetGroupArn' --output text)

echo "✅ Target group created: $TARGET_GROUP_ARN"

# Create listener
echo "🔊 Creating load balancer listener..."
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN \
    --region $AWS_REGION

# Create ECS service
echo "🚀 Creating ECS service..."
aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition datasafe:1 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_ARRAY[0]},${SUBNET_ARRAY[1]}],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$TARGET_GROUP_ARN,containerName=datasafe,containerPort=8000" \
    --region $AWS_REGION

# Clean up temporary file
rm -f task-definition.json

echo "🎉 AWS setup complete!"
echo ""
echo "📋 Summary:"
echo "  - ECS Cluster: $CLUSTER_NAME"
echo "  - ECS Service: $SERVICE_NAME"
echo "  - Load Balancer: $ALB_ARN"
echo "  - Security Group: $SECURITY_GROUP_ID"
echo ""
echo "🚀 Next steps:"
echo "  1. Run: ./deploy-ecs.sh"
echo "  2. Get load balancer DNS: aws elbv2 describe-load-balancers --names datasafe-alb --region $AWS_REGION --query 'LoadBalancers[0].DNSName' --output text"
echo "  3. Access your application at the load balancer DNS"
echo ""
echo "📊 Monitor deployment:"
echo "  aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION" 