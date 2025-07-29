# 🚀 Deploy MediaMap to Your Existing AWS Instance

## Your AWS Details
- **Instance ID:** `i-09086b9d6aaf71e21`
- **Public IP:** `54.87.58.143`
- **Account ID:** `498787422066`
- **Region:** `us-east-1`

## 🎯 Quick Deployment (Recommended)

### Step 1: Install AWS CLI (if not already installed)
```bash
# macOS
brew install awscli

# Or download from AWS website
# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
```

### Step 2: Place Your SSH Key
Put your `.pem` key file in the project directory. The script will automatically find it.

### Step 3: Deploy with One Command
```bash
./quick-deploy.sh
```

That's it! The script will:
- ✅ Verify your AWS credentials
- ✅ Check and start your instance if needed
- ✅ Open necessary ports (80, 8000)
- ✅ Install Docker and dependencies
- ✅ Deploy MediaMap application
- ✅ Verify the deployment

## 🔍 Manual Deployment (if needed)

If the quick deploy doesn't work, you can deploy manually:

```bash
# 1. Load AWS configuration
source ./aws-config.sh

# 2. Deploy to your instance
./deploy-to-existing-instance.sh
```

## 📋 What Gets Installed

- **Docker & Docker Compose** - Container management
- **Git** - Code repository access
- **MediaMap Application** - Your Flask app
- **Nginx** - Web server and reverse proxy
- **SSL Support** - HTTPS ready

## 🌐 Access Your Application

Once deployed, your MediaMap will be available at:
**http://54.87.58.143**

## 🔍 Monitoring Commands

```bash
# Check application status
ssh -i your-key.pem ubuntu@54.87.58.143 'docker-compose ps'

# View application logs
ssh -i your-key.pem ubuntu@54.87.58.143 'docker-compose logs -f'

# Check system resources
ssh -i your-key.pem ubuntu@54.87.58.143 'htop'

# Restart application
ssh -i your-key.pem ubuntu@54.87.58.143 'cd /opt/mediamap && docker-compose restart'
```

## 🔧 Environment Configuration

The deployment will create a `.env` file on your instance. You'll need to edit it with your actual values:

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@54.87.58.143

# Edit environment file
nano /opt/mediamap/.env
```

Required values:
```bash
SECRET_KEY=your-super-secret-production-key
OPENAI_API_KEY=your-openai-api-key
FLASK_ENV=production
```

## 🚨 Troubleshooting

### Application won't start
```bash
# Check logs
ssh -i your-key.pem ubuntu@54.87.58.143 'docker-compose logs'

# Check if ports are open
ssh -i your-key.pem ubuntu@54.87.58.143 'sudo netstat -tlnp | grep :8000'
```

### Can't access the website
```bash
# Check if instance is running
aws ec2 describe-instances --instance-ids i-09086b9d6aaf71e21

# Check security group rules
aws ec2 describe-security-groups --group-ids $(aws ec2 describe-instances --instance-ids i-09086b9d6aaf71e21 --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)
```

### SSH connection issues
```bash
# Check key permissions
chmod 400 your-key.pem

# Test SSH connection
ssh -i your-key.pem ubuntu@54.87.58.143 'echo "SSH working!"'
```

## 💰 Cost Information

Your current setup:
- **Instance Type:** Check with `aws ec2 describe-instances --instance-ids i-09086b9d6aaf71e21`
- **Estimated Cost:** ~$20-80/month depending on instance size
- **Storage:** Included in instance cost

## 🎯 Next Steps

1. **Deploy:** Run `./quick-deploy.sh`
2. **Configure:** Edit the `.env` file with your API keys
3. **Test:** Visit http://54.87.58.143
4. **Monitor:** Use the monitoring commands above
5. **Custom Domain:** Point your domain to 54.87.58.143

## 🆘 Need Help?

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the logs: `ssh -i your-key.pem ubuntu@54.87.58.143 'docker-compose logs'`
3. Verify AWS credentials: `aws sts get-caller-identity`

---

**Ready to deploy? Run `./quick-deploy.sh` and let's get MediaMap running! 🚀** 