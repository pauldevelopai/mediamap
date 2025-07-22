# MediaMap Production Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- 10GB free disk space

### 1. Clone and Deploy
```bash
git clone https://github.com/pauldevelopai/mediamap.git
cd mediamap
./deploy.sh
```

### 2. Access Your Application
- **Main Application:** http://localhost
- **Health Check:** http://localhost:8000/health
- **Admin Panel:** http://localhost/admin

## 🔧 Manual Deployment

### Using Docker Compose
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Docker Only
```bash
# Build the image
docker build -t mediamap .

# Run the container
docker run -d -p 8000:8000 --name mediamap-app mediamap
```

## 🌐 Production Deployment Options

### Option 1: Cloud Deployment (Recommended)

#### AWS EC2
```bash
# Launch EC2 instance (t3.medium or larger)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy
git clone https://github.com/pauldevelopai/mediamap.git
cd mediamap
./deploy.sh
```

#### Google Cloud Run
```bash
# Build and push to Google Container Registry
docker build -t gcr.io/YOUR_PROJECT/mediamap .
docker push gcr.io/YOUR_PROJECT/mediamap

# Deploy to Cloud Run
gcloud run deploy mediamap \
  --image gcr.io/YOUR_PROJECT/mediamap \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 2: VPS Deployment
```bash
# On your VPS
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker

# Deploy
git clone https://github.com/pauldevelopai/mediamap.git
cd mediamap
./deploy.sh
```

## 🔒 Security Configuration

### SSL/HTTPS Setup
1. Obtain SSL certificates (Let's Encrypt recommended)
2. Update `nginx.conf` with your certificate paths
3. Uncomment SSL configuration lines
4. Restart services: `docker-compose restart`

### Environment Variables
Create a `.env` file with:
```bash
SECRET_KEY=your-super-secret-production-key
OPENAI_API_KEY=your-openai-api-key
FLASK_ENV=production
```

## 📊 Monitoring and Maintenance

### Health Checks
- **Endpoint:** `/health`
- **Expected Response:** `{"status": "healthy", "model_loaded": true}`

### Logs
```bash
# View application logs
docker-compose logs -f mediamap

# View nginx logs
docker-compose logs -f nginx
```

### Database Backup
```bash
# Backup SQLite database
docker exec mediamap_mediamap_1 sqlite3 /app/instance/media_analysis.db ".backup /app/backup.db"
docker cp mediamap_mediamap_1:/app/backup.db ./backup_$(date +%Y%m%d).db
```

### Updates
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using port 8000
sudo lsof -i :8000

# Kill the process or change port in docker-compose.yml
```

#### Out of Memory
```bash
# Check memory usage
docker stats

# Increase swap space or upgrade instance
```

#### Model Loading Issues
```bash
# Check model files
docker exec mediamap_mediamap_1 ls -la /app/backend/training/models/deployment/model/

# Rebuild without cache
docker-compose build --no-cache mediamap
```

### Performance Optimization

#### For High Traffic
1. Increase worker processes in `gunicorn.conf.py`
2. Add Redis for session storage
3. Use PostgreSQL instead of SQLite
4. Enable CDN for static files

#### Resource Limits
```yaml
# In docker-compose.yml
services:
  mediamap:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## 📞 Support

For deployment issues:
1. Check logs: `docker-compose logs`
2. Verify health: `curl http://localhost:8000/health`
3. Check resource usage: `docker stats`
4. Review this documentation

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to server
        run: |
          ssh user@your-server "cd /path/to/mediamap && git pull && ./deploy.sh"
``` 