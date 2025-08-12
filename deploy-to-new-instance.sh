#!/bin/bash
set -e

# New upgraded Lightsail instance details
INSTANCE_IP="35.176.169.218"
INSTANCE_NAME="DATASAFEUPGRADE"
KEY_FILE="lightsail-key.pem"

echo "🚀 Deploying DataSafe to upgraded Lightsail instance..."
echo "📋 Instance: $INSTANCE_NAME"
echo "🌐 IP: $INSTANCE_IP"
echo "💾 Specs: 2GB RAM, 2 vCPUs, 60GB SSD"

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "❌ SSH key file not found: $KEY_FILE"
    exit 1
fi

# Test connectivity
echo "🔍 Testing connectivity..."
if ! ping -c 1 $INSTANCE_IP > /dev/null 2>&1; then
    echo "⚠️  Cannot ping instance (firewall may block ICMP)"
    echo "   Proceeding with SSH test..."
fi

# Test SSH connection
echo "🔑 Testing SSH connection..."
if ! ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$INSTANCE_IP 'echo "SSH connection successful"' 2>/dev/null; then
    echo "❌ SSH connection failed!"
    echo "💡 Please ensure:"
    echo "   1. Instance is fully started (green status)"
    echo "   2. SSH key is correct"
    echo "   3. Firewall allows SSH (port 22)"
    exit 1
fi

echo "✅ SSH connection successful!"

# Create deployment script for remote execution
cat > remote-deploy-upgraded.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting DataSafe deployment on upgraded instance..."
echo "📦 Updating system packages..."

# Update system
sudo apt update && sudo apt upgrade -y

echo "🐳 Installing Docker..."
# Install Docker
sudo apt install -y docker.io docker-compose curl git htop

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

echo "📁 Setting up application directory..."
# Create app directory
sudo mkdir -p /opt/datasafe
sudo chown ubuntu:ubuntu /opt/datasafe
cd /opt/datasafe

echo "📥 Cloning repository..."
# Clone the repository
git clone https://github.com/pauldevelopai/datasafe.git . || echo "Repository already exists, updating..." && git pull

echo "📝 Creating environment file..."
# Create .env file
cat > .env << 'ENVEOF'
FLASK_ENV=production
FLASK_APP=backend/app.py
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///instance/datasafe.db
ENVEOF

echo "🔨 Installing Python dependencies..."
# Install Python dependencies
python3 -m pip install --user -r requirements.txt

echo "🚀 Starting application..."
# Start the application
nohup python3 backend/app.py --host=0.0.0.0 --port=8000 > app.log 2>&1 &

echo "✅ DataSafe deployment complete!"
echo "🌐 Access your app at: http://$(curl -s ifconfig.me):8000"
echo "📋 Check logs with: tail -f /opt/datasafe/app.log"
EOF

# Copy deployment script to instance
echo "📤 Copying deployment script..."
scp -i $KEY_FILE -o StrictHostKeyChecking=no remote-deploy-upgraded.sh ubuntu@$INSTANCE_IP:/tmp/

# Execute deployment script on instance
echo "🔨 Running deployment script..."
ssh -i $KEY_FILE -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "chmod +x /tmp/remote-deploy-upgraded.sh && /tmp/remote-deploy-upgraded.sh"

echo "🎉 Deployment complete!"
echo "🌐 Your DataSafe app should be available at: http://$INSTANCE_IP:8000"
echo "📋 To check status: ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'ps aux | grep python'" 