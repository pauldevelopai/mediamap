# 🚀 MediaMap AWS Quick Start

## Choose Your Deployment Path

### 🎯 Option 1: ECS (Recommended for Production)
**Best for:** Auto-scaling, managed container orchestration, production workloads
**Cost:** ~$50-150/month
**Time:** 15-20 minutes

### 💰 Option 2: EC2 (Most Cost-Effective)
**Best for:** Full control, consistent workloads, learning AWS
**Cost:** ~$20-80/month
**Time:** 10-15 minutes

### ⚡ Option 3: App Runner (Simplest)
**Best for:** Quick deployment, managed service
**Cost:** ~$30-100/month
**Time:** 5-10 minutes

---

## 🚀 ECS Deployment (Recommended)

### Prerequisites
1. AWS account with appropriate permissions
2. AWS CLI installed and configured
3. Docker installed locally

### Step 1: Configure AWS CLI
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
# Enter your output format (json)
```

### Step 2: Set Up AWS Resources
```bash
./setup-aws.sh
```

### Step 3: Deploy Your Application
```bash
./deploy-ecs.sh
```

### Step 4: Get Your Application URL
```bash
aws elbv2 describe-load-balancers --names mediamap-alb --region us-east-1 --query 'LoadBalancers[0].DNSName' --output text
```

---

## 💰 EC2 Deployment (Cost-Effective)

### Prerequisites
1. AWS account with appropriate permissions
2. SSH key pair created in AWS

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
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mediamap-server}]'
```

### Step 2: SSH and Deploy
```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker and dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu

# Clone and deploy
git clone https://github.com/pauldevelopai/mediamap.git
cd mediamap
./deploy-ec2.sh
```

---

## ⚡ App Runner Deployment (Simplest)

### Step 1: Create App Runner Service
```bash
aws apprunner create-service \
  --service-name mediamap \
  --source-configuration '{
    "RepositoryUrl": "https://github.com/pauldevelopai/mediamap",
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

### Step 2: Get Your Application URL
```bash
aws apprunner describe-service --service-name mediamap --query 'Service.ServiceUrl' --output text
```

---

## 🔐 Environment Variables

Create a `.env` file with your configuration:

```bash
SECRET_KEY=your-super-secret-production-key
OPENAI_API_KEY=your-openai-api-key
FLASK_ENV=production
```

## 📊 Monitoring

### Check Application Health
```bash
# ECS
aws ecs describe-services --cluster mediamap-cluster --services mediamap-service --region us-east-1

# EC2
curl -f http://localhost:8000/health

# App Runner
curl -f https://your-app-runner-url/health
```

### View Logs
```bash
# ECS
aws logs tail /ecs/mediamap --follow --region us-east-1

# EC2
docker-compose logs -f

# App Runner
aws apprunner describe-service --service-name mediamap --query 'Service.ServiceUrl' --output text
```

## 💰 Cost Optimization Tips

1. **Use Spot Instances** for ECS (save 60-90%)
2. **Reserved Instances** for EC2 (save 30-60%)
3. **Right-size instances** based on actual usage
4. **Set up auto-scaling** to scale down during low usage
5. **Use CloudWatch** to monitor costs

## 🚨 Troubleshooting

### Common Issues

**Application won't start:**
```bash
# Check logs
docker-compose logs
aws logs tail /ecs/mediamap --region us-east-1
```

**Health check failing:**
```bash
# Check if port 8000 is accessible
curl -f http://localhost:8000/health
```

**Out of memory:**
```bash
# Increase memory in task definition or instance size
# For ECS: Update task definition with more memory
# For EC2: Upgrade to larger instance type
```

## 📞 Need Help?

1. Check the detailed guide: `aws-deployment-guide.md`
2. Review logs for error messages
3. Verify AWS credentials and permissions
4. Ensure all required services are running

## 🎯 Next Steps

1. **Set up a custom domain** with Route 53
2. **Configure SSL certificates** with ACM
3. **Set up monitoring** with CloudWatch
4. **Configure auto-scaling** for production
5. **Set up backup strategies** for your data

---

**Ready to deploy? Choose your path and let's get MediaMap running on AWS! 🚀** 