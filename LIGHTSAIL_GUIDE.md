# 🚀 DataSafe on AWS Lightsail

## Why Lightsail?

Lightsail is perfect for DataSafe because:
- ✅ **Simpler than EC2** - Managed VPS with easy setup
- ✅ **Predictable pricing** - $3.50/month for nano instance
- ✅ **Built-in networking** - Automatic firewall and DNS
- ✅ **Easy scaling** - Upgrade bundles as needed
- ✅ **Integrated management** - Console, CLI, and API

## 🎯 Quick Deployment

### One-Command Deployment
```bash
./deploy-lightsail.sh
```

This will:
- Create a new Lightsail instance (or use existing)
- Install Docker and dependencies
- Deploy DataSafe application
- Configure networking
- Test the deployment

## 📊 Instance Options

### Nano Bundle (Recommended for starting)
- **Specs:** 512MB RAM, 1 vCPU, 20GB SSD
- **Cost:** $3.50/month
- **Perfect for:** Development, testing, low traffic

### Micro Bundle (For production)
- **Specs:** 1GB RAM, 1 vCPU, 40GB SSD  
- **Cost:** $7/month
- **Perfect for:** Production workloads, higher traffic

### Small Bundle (For scaling)
- **Specs:** 2GB RAM, 1 vCPU, 60GB SSD
- **Cost:** $10/month
- **Perfect for:** Multiple users, AI model hosting

## 🔧 Manual Setup (Alternative)

### Step 1: Create Lightsail Instance
```bash
# Create instance
aws lightsail create-instances \
  --instance-names datasafe-server \
  --availability-zone us-east-1a \
  --blueprint-id ubuntu_22_04 \
  --bundle-id nano_2_0
```

### Step 2: Open Ports
```bash
# Open HTTP, HTTPS, and application ports
aws lightsail open-instance-public-ports \
  --instance-name datasafe-server \
  --port-info fromPort=80,toPort=80,protocol=tcp

aws lightsail open-instance-public-ports \
  --instance-name datasafe-server \
  --port-info fromPort=443,toPort=443,protocol=tcp

aws lightsail open-instance-public-ports \
  --instance-name datasafe-server \
  --port-info fromPort=8000,toPort=8000,protocol=tcp
```

### Step 3: Get Instance Details
```bash
# Get IP address
aws lightsail get-instances \
  --instance-names datasafe-server \
  --query 'instances[0].publicIpAddress' \
  --output text

# Get SSH key
aws lightsail download-default-key-pair --output text > LightsailDefaultKey-us-east-1.pem
chmod 400 LightsailDefaultKey-us-east-1.pem
```

### Step 4: Deploy Application
```bash
# SSH into instance
ssh -i LightsailDefaultKey-us-east-1.pem ubuntu@YOUR_INSTANCE_IP

# Install Docker
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu

# Clone and deploy DataSafe
cd /opt
sudo mkdir datasafe
sudo chown ubuntu:ubuntu datasafe
cd datasafe
git clone https://github.com/pauldevelopai/datasafe.git .

# Create environment file
cat > .env << EOF
SECRET_KEY=your-super-secret-production-key
OPENAI_API_KEY=your-openai-api-key
FLASK_ENV=production
EOF

# Build and start
docker-compose build --no-cache
docker-compose up -d
```

## 🌐 Access Your Application

Once deployed, your DataSafe will be available at:
- **Main site:** `http://YOUR_INSTANCE_IP`
- **Direct app:** `http://YOUR_INSTANCE_IP:8000`

## 🔍 Management Commands

### Check Instance Status
```bash
aws lightsail get-instances --instance-names datasafe-server
```

### Start/Stop Instance
```bash
# Start
aws lightsail start-instance --instance-name datasafe-server

# Stop
aws lightsail stop-instance --instance-name datasafe-server
```

### SSH Access
```bash
ssh -i LightsailDefaultKey-us-east-1.pem ubuntu@YOUR_INSTANCE_IP
```

### Application Management
```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Restart application
docker-compose restart

# Update application
git pull origin main
docker-compose build --no-cache
docker-compose up -d
```

## 💰 Cost Breakdown

### Monthly Costs
- **Nano instance:** $3.50
- **Data transfer:** First 1TB free
- **Storage:** Included in bundle
- **Total:** ~$3.50-5.00/month

### Scaling Costs
- **Micro (1GB):** $7/month
- **Small (2GB):** $10/month
- **Medium (4GB):** $20/month

## 🔧 Configuration

### Environment Variables
Edit `/opt/datasafe/.env` on your instance:
```bash
SECRET_KEY=your-super-secret-production-key
OPENAI_API_KEY=your-openai-api-key
FLASK_ENV=production
```

### Custom Domain
1. Point your domain to the Lightsail IP
2. Add DNS record: `A @ YOUR_INSTANCE_IP`
3. Configure SSL with Let's Encrypt

### SSL Certificate
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com
```

## 📊 Monitoring

### Lightsail Console
- **Dashboard:** Monitor CPU, memory, network
- **Metrics:** Real-time performance data
- **Alarms:** Set up notifications

### Application Monitoring
```bash
# Check resource usage
htop

# Monitor Docker containers
docker stats

# Check application logs
docker-compose logs -f
```

## 🚨 Troubleshooting

### Instance Won't Start
```bash
# Check instance state
aws lightsail get-instances --instance-names datasafe-server

# Check console output
aws lightsail get-instance-access-details --instance-name datasafe-server
```

### Application Won't Load
```bash
# Check if ports are open
aws lightsail get-instance-port-states --instance-name datasafe-server

# Check application logs
ssh -i LightsailDefaultKey-us-east-1.pem ubuntu@YOUR_INSTANCE_IP 'docker-compose logs'
```

### Out of Memory
```bash
# Upgrade to larger bundle
aws lightsail create-instances \
  --instance-names datasafe-server-new \
  --availability-zone us-east-1a \
  --blueprint-id ubuntu_22_04 \
  --bundle-id micro_2_0
```

## 🔄 Backup & Recovery

### Create Snapshot
```bash
aws lightsail create-instance-snapshot \
  --instance-snapshot-name datasafe-backup-$(date +%Y%m%d) \
  --instance-name datasafe-server
```

### Restore from Snapshot
```bash
aws lightsail create-instances-from-snapshot \
  --instance-names datasafe-restored \
  --availability-zone us-east-1a \
  --bundle-id nano_2_0 \
  --source-snapshot-name datasafe-backup-20241201
```

## 🎯 Next Steps

1. **Deploy:** Run `./deploy-lightsail.sh`
2. **Configure:** Edit environment variables
3. **Test:** Visit your application URL
4. **Monitor:** Set up monitoring and alerts
5. **Scale:** Upgrade bundle as needed
6. **Domain:** Add custom domain and SSL

## 💡 Pro Tips

- **Use nano bundle** for development/testing
- **Upgrade to micro** for production
- **Set up snapshots** for backup
- **Monitor costs** in Lightsail console
- **Use Lightsail DNS** for easy domain management
- **Enable notifications** for instance events

---

**Ready to deploy? Run `./deploy-lightsail.sh` and get DataSafe running on Lightsail! 🚀** 