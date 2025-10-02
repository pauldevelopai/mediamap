# 🚀 Deploy AIMAP to AWS Lightsail

## Quick Deployment (5 minutes)

### Prerequisites
- AWS CLI configured with credentials
- SSH key pair for Lightsail

### Step 1: Configure AWS Credentials
```bash
# Set your AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### Step 2: Deploy to Lightsail
```bash
# Run the deployment script
./deploy-lightsail.sh
```

### Step 3: Access Your Application
- **Main URL**: http://your-instance-ip
- **Direct Flask**: http://your-instance-ip:8000
- **SSH Access**: `ssh -i your-key.pem ubuntu@your-instance-ip`

## What Gets Deployed

✅ **AIMAP Core Application**
- Multi-sector AI adoption tracking
- Intelligent scoring engine
- Predictive analytics with ML models
- Risk assessment and ROI estimation

✅ **Machine Learning Features**
- AI adoption trajectory prediction
- Risk scoring for organizations
- Investment ROI calculations
- Sector insights and benchmarking

✅ **Security & DataSafe Integration**
- Threat intelligence dashboard
- Security risk assessment
- Data protection monitoring

✅ **Reporting & Analytics**
- PPTX report generation
- PDF export capabilities
- Interactive dashboards
- Peer benchmarking

## Cost
- **Lightsail Instance**: ~$3.50/month (nano bundle)
- **Domain**: Optional (~$12/year)
- **Total**: ~$4/month for full AI intelligence platform

## Management Commands

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Check application status
cd /opt/aimap
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

## Environment Variables

Create a `.env` file on your Lightsail instance:
```bash
SECRET_KEY=your-super-secret-production-key
OPENAI_API_KEY=your-openai-api-key
FLASK_ENV=production
DATABASE_URL=sqlite:///./instance/media_analysis.db
```

## Troubleshooting

### Application Not Starting
```bash
# Check Docker status
sudo systemctl status docker

# Check application logs
docker-compose logs aimap

# Restart Docker
sudo systemctl restart docker
```

### ML Models Not Working
```bash
# Initialize ML models
curl -X POST http://your-instance-ip:8000/api/ml/initialize

# Check model status
curl http://your-instance-ip:8000/api/ml/status
```

### Database Issues
```bash
# Run migrations
cd /opt/aimap
python -m alembic upgrade head

# Seed demo data
python scripts/aimap_cli.py seed-demo --sector Media --n 10
```

## Next Steps

1. **Customize Configuration**: Update environment variables
2. **Add Domain**: Point your domain to the Lightsail IP
3. **Set Up SSL**: Configure HTTPS with Let's Encrypt
4. **Backup Strategy**: Set up automated database backups
5. **Monitoring**: Add application monitoring and alerts

## Support

For issues or questions:
- Check application logs: `docker-compose logs`
- Review this deployment guide
- Check AIMAP documentation in README.md
