# DataSafe AWS Deployment Guide

## 🚀 Quick Start Options

### Option 1: AWS ECS (Elastic Container Service) - Recommended
**Best for:** Production workloads, auto-scaling, managed container orchestration
**Cost:** ~$50-150/month depending on usage

### Option 2: AWS EC2 with Docker
**Best for:** Full control, cost-effective for consistent workloads
**Cost:** ~$20-80/month depending on instance size

### Option 3: AWS App Runner
**Best for:** Simple deployment, managed service
**Cost:** ~$30-100/month depending on usage

## 📋 Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Docker** installed locally
4. **Domain name** (optional but recommended)

## 🔧 Option 1: AWS ECS Deployment (Recommended)

### Step 1: Create ECR Repository
```bash
# Create ECR repository
aws ecr create-repository --repository-name datasafe --region us-east-1

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com

# Tag and push image
docker build -t datasafe .
docker tag datasafe:latest $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com/datasafe:latest
docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com/datasafe:latest
```

### Step 2: Create ECS Cluster
```bash
# Create cluster
aws ecs create-cluster --cluster-name datasafe-cluster --region us-east-1
```

### Step 3: Create Task Definition
Create `task-definition.json`:
```json
{
  "family": "datasafe",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "datasafe",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/datasafe:latest",
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
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:datasafe/secret-key"
        },
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:datasafe/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/datasafe",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Step 4: Create Application Load Balancer
```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name datasafe-alb \
  --subnets subnet-12345678 subnet-87654321 \
  --security-groups sg-12345678 \
  --region us-east-1

# Create target group
aws elbv2 create-target-group \
  --name datasafe-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-12345678 \
  --target-type ip \
  --region us-east-1

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:YOUR_ACCOUNT_ID:loadbalancer/app/datasafe-alb/1234567890abcdef \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:YOUR_ACCOUNT_ID:targetgroup/datasafe-tg/1234567890abcdef
```

### Step 5: Create ECS Service
```bash
# Create service
aws ecs create-service \
  --cluster datasafe-cluster \
  --service-name datasafe-service \
  --task-definition datasafe:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678,subnet-87654321],securityGroups=[sg-12345678],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:YOUR_ACCOUNT_ID:targetgroup/datasafe-tg/1234567890abcdef,containerName=datasafe,containerPort=8000"
```

## 🔧 Option 2: EC2 with Docker (Cost-Effective)

### Step 1: Launch EC2 Instance
```bash
# Launch t3.medium instance with Ubuntu 22.04
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --count 1 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=datasafe-server}]'
```

### Step 2: Configure EC2 Instance
```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu

# Install AWS CLI
sudo apt install -y awscli

# Logout and login again for group changes
exit
ssh -i your-key.pem ubuntu@your-instance-ip
```

### Step 3: Deploy Application
```bash
# Clone repository
git clone https://github.com/pauldevelopai/datasafe.git
cd datasafe

# Create environment file
cat > .env << EOF
SECRET_KEY=your-super-secret-production-key
OPENAI_API_KEY=your-openai-api-key
FLASK_ENV=production
EOF

# Build and run
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

### Step 4: Configure Domain and SSL
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

## 🔧 Option 3: AWS App Runner (Simplest)

### Step 1: Prepare Application
```bash
# Create apprunner.yaml
cat > apprunner.yaml << EOF
version: 1.0
runtime: python3
build:
  commands:
    build:
      - echo "Building DataSafe application"
      - pip install -r requirements.txt
run:
  runtime-version: 3.12
  command: gunicorn --config gunicorn.conf.py backend.app:app
  network:
    port: 8000
    env: PORT
EOF
```

### Step 2: Deploy to App Runner
```bash
# Create App Runner service
aws apprunner create-service \
  --service-name datasafe \
  --source-configuration '{
    "RepositoryUrl": "https://github.com/pauldevelopai/datasafe",
    "SourceCodeVersion": {
      "Type": "BRANCH",
      "Value": "main"
    },
    "CodeConfiguration": {
      "ConfigurationSource": "API",
      "CodeConfigurationValues": {
        "Runtime": "PYTHON_3",
        "BuildCommand": "pip install -r requirements.txt",
        "StartCommand": "gunicorn --config gunicorn.conf.py backend.app:app",
        "Port": "8000"
      }
    }
  }' \
  --instance-configuration '{
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  }'
```

## 🔐 Security Setup

### Step 1: Create IAM Roles
```bash
# Create ECS task execution role
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
  }'

# Attach policies
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

### Step 2: Store Secrets
```bash
# Create secrets
aws secretsmanager create-secret \
  --name datasafe/secret-key \
  --description "DataSafe Flask secret key" \
  --secret-string "your-super-secret-production-key"

aws secretsmanager create-secret \
  --name datasafe/openai-key \
  --description "OpenAI API key" \
  --secret-string "your-openai-api-key"
```

### Step 3: Configure Security Groups
```bash
# Create security group
aws ec2 create-security-group \
  --group-name datasafe-sg \
  --description "Security group for DataSafe application"

# Allow HTTP/HTTPS
aws ec2 authorize-security-group-ingress \
  --group-name datasafe-sg \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-name datasafe-sg \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# Allow SSH (for EC2 option)
aws ec2 authorize-security-group-ingress \
  --group-name datasafe-sg \
  --protocol tcp \
  --port 22 \
  --cidr YOUR_IP/32
```

## 📊 Monitoring and Scaling

### CloudWatch Setup
```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
  --dashboard-name DataSafe-Dashboard \
  --dashboard-body '{
    "widgets": [
      {
        "type": "metric",
        "properties": {
          "metrics": [
            ["AWS/ECS", "CPUUtilization", "ServiceName", "datasafe-service", "ClusterName", "datasafe-cluster"]
          ],
          "period": 300,
          "stat": "Average",
          "region": "us-east-1",
          "title": "ECS CPU Utilization"
        }
      }
    ]
  }'
```

### Auto Scaling (ECS)
```bash
# Create auto scaling target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/datasafe-cluster/datasafe-service \
  --min-capacity 1 \
  --max-capacity 10

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/datasafe-cluster/datasafe-service \
  --policy-name datasafe-cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    }
  }'
```

## 💰 Cost Optimization

### Reserved Instances (EC2)
```bash
# Purchase reserved instance for 1-year term
aws ec2 describe-reserved-instances-offerings \
  --instance-type t3.medium \
  --offering-type All Upfront \
  --product-description "Linux/UNIX" \
  --max-duration 31536000
```

### Spot Instances (ECS)
```bash
# Use spot instances for cost savings
aws ecs create-service \
  --cluster datasafe-cluster \
  --service-name datasafe-spot-service \
  --task-definition datasafe:1 \
  --desired-count 2 \
  --launch-type FARGATE_SPOT \
  --capacity-provider-strategy 'capacityProvider=FARGATE_SPOT,weight=1'
```

## 🚀 Deployment Scripts

### Automated ECS Deployment
Create `deploy-ecs.sh`:
```bash
#!/bin/bash
set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPO="datasafe"
CLUSTER_NAME="datasafe-cluster"
SERVICE_NAME="datasafe-service"

# Build and push image
echo "Building Docker image..."
docker build -t $ECR_REPO .

echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com

echo "Tagging and pushing image..."
docker tag $ECR_REPO:latest $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# Update ECS service
echo "Updating ECS service..."
aws ecs update-service \
  --cluster $CLUSTER_NAME \
  --service $SERVICE_NAME \
  --force-new-deployment \
  --region $AWS_REGION

echo "Deployment complete! Service is updating..."
```

### EC2 Deployment Script
Create `deploy-ec2.sh`:
```bash
#!/bin/bash
set -e

# Pull latest code
git pull origin main

# Stop existing containers
docker-compose down

# Build new image
docker-compose build --no-cache

# Start services
docker-compose up -d

# Check health
sleep 30
curl -f http://localhost:8000/health || exit 1

echo "Deployment complete!"
```

## 🔄 CI/CD Pipeline

### GitHub Actions for ECS
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to AWS ECS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build, tag, and push image to Amazon ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: datasafe
          IMAGE_TAG: latest
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster datasafe-cluster --service datasafe-service --force-new-deployment
```

## 📞 Next Steps

1. **Choose your deployment option** based on your needs:
   - ECS: Best for production, auto-scaling
   - EC2: Most cost-effective, full control
   - App Runner: Simplest deployment

2. **Set up your AWS environment**:
   - Create AWS account if you don't have one
   - Install and configure AWS CLI
   - Create necessary IAM roles and security groups

3. **Deploy your application**:
   - Follow the step-by-step guide for your chosen option
   - Configure environment variables and secrets
   - Set up monitoring and logging

4. **Configure domain and SSL**:
   - Point your domain to the AWS service
   - Set up SSL certificates
   - Configure CDN if needed

5. **Monitor and optimize**:
   - Set up CloudWatch dashboards
   - Configure auto-scaling
   - Monitor costs and performance

## 💡 Pro Tips

- **Start with EC2** if you're new to AWS - it's the most straightforward
- **Use ECS** for production workloads that need auto-scaling
- **Consider App Runner** if you want the simplest deployment experience
- **Set up CloudWatch alarms** for monitoring
- **Use AWS Secrets Manager** for sensitive configuration
- **Enable CloudTrail** for audit logging
- **Set up backup strategies** for your data

Need help with any specific step? Let me know which deployment option you'd like to pursue! 