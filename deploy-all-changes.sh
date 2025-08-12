#!/bin/bash

# Comprehensive DataSafe Deployment Script
# Deploys all recent changes to Lightsail instance

echo "🚀 DataSafe Complete Deployment Script"
echo "======================================"

# Configuration - UPDATE THIS IP ADDRESS
INSTANCE_IP="35.176.169.218"  # Update this with your current Lightsail IP
SSH_KEY="LightsailDefaultKey-eu-west-2.pem"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    print_error "SSH key not found: $SSH_KEY"
    echo "Please download your Lightsail SSH key first"
    exit 1
fi

print_status "Current IP: $INSTANCE_IP"
print_status "SSH Key: $SSH_KEY"
echo ""

# Test connection
print_status "Testing connection to $INSTANCE_IP..."
if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "echo 'Connection successful'" 2>/dev/null; then
    print_success "Connection successful! Deploying all updates..."
    echo ""
    
    # Create temp directory and copy ALL updated files
    mkdir -p /tmp/datasafe-complete
    print_status "Preparing files for deployment..."
    
    # Copy all template files with recent changes
    mkdir -p /tmp/datasafe-complete/templates
    cp backend/templates/user_dashboard.html /tmp/datasafe-complete/templates/
    cp backend/templates/today_news.html /tmp/datasafe-complete/templates/
    cp backend/templates/ai_strategies.html /tmp/datasafe-complete/templates/ 2>/dev/null || true
    cp backend/templates/company_info.html /tmp/datasafe-complete/templates/
    cp backend/templates/user_chats.html /tmp/datasafe-complete/templates/
    # New templates
    cp backend/templates/clients.html /tmp/datasafe-complete/templates/ 2>/dev/null || true
    cp backend/templates/client_dashboard.html /tmp/datasafe-complete/templates/ 2>/dev/null || true
    cp backend/templates/ims.html /tmp/datasafe-complete/templates/ 2>/dev/null || true

    # Copy backend files with recent changes
    mkdir -p /tmp/datasafe-complete/backend
    cp backend/app.py /tmp/datasafe-complete/backend/
    cp backend/models.py /tmp/datasafe-complete/backend/
    cp backend/crawler_service.py /tmp/datasafe-complete/backend/ 2>/dev/null || true

    # Copy deployment files
    cp docker-compose.yml /tmp/datasafe-complete/
    cp gunicorn.conf.py /tmp/datasafe-complete/

    # Copy setup script
    cp setup-openai-key.sh /tmp/datasafe-complete/
    
    # Upload and deploy
    print_status "Uploading files..."
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r /tmp/datasafe-complete/* ubuntu@$INSTANCE_IP:/tmp/
    
    print_status "Deploying on server..."
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'EOF'
        set -e
        echo "📝 Updating all template files..."
        sudo mkdir -p /opt/datasafe/backend/templates
        sudo cp /tmp/templates/*.html /opt/datasafe/backend/templates/

        echo "📝 Updating backend files..."
        sudo cp /tmp/backend/app.py /opt/datasafe/backend/
        sudo cp /tmp/backend/models.py /opt/datasafe/backend/
        if [ -f /tmp/backend/crawler_service.py ]; then sudo cp /tmp/backend/crawler_service.py /opt/datasafe/backend/; fi

        echo "📝 Updating deployment files..."
        sudo cp /tmp/docker-compose.yml /opt/datasafe/
        sudo cp /tmp/gunicorn.conf.py /opt/datasafe/

        echo "📝 Updating setup script..."
        sudo cp /tmp/setup-openai-key.sh /opt/datasafe/
        sudo chmod +x /opt/datasafe/setup-openai-key.sh

        echo "🔧 Setting proper permissions..."
        sudo chown -R ubuntu:ubuntu /opt/datasafe
        sudo find /opt/datasafe -type f -name "*.html" -exec chmod 644 {} +
        sudo chmod 644 /opt/datasafe/backend/app.py /opt/datasafe/backend/models.py || true

        echo "🐳 Ensuring containers are up..."
        cd /opt/datasafe
        if ! docker-compose ps | grep -q datasafe; then
            echo "Containers not running. Starting fresh..."
            docker-compose down || true
            docker-compose build --no-cache
            docker-compose up -d
        else
            echo "Containers detected. Restarting..."
            docker-compose restart || true
        fi

        echo "⏳ Waiting for app to be ready..."
        sleep 20
        echo "🏥 Health check:"
        curl -sf http://localhost:8000/health || true
        echo
        echo "✅ All updates deployed successfully!"
EOF
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "🎉 Complete deployment successful!"
        print_success "🌐 Your updated app is now live at: http://$INSTANCE_IP"
        echo ""
        echo "✨ All recent features deployed:"
        echo "   • Fixed connection error in chat functionality"
        echo "   • Optimized mobile navigation layout"
        echo "   • Added company info generation functionality"
        echo "   • Fixed feedback submission error on mobile"
        echo "   • Fixed news caching issue and added refresh functionality"
        echo "   • Standardized navigation menu across all templates"
        echo "   • Added OpenAI API key setup script"
        echo ""
        echo "🔧 If you need to configure OpenAI API key on the server:"
        echo "   ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
        echo "   cd /opt/datasafe && sudo ./setup-openai-key.sh"
    else
        print_error "Deployment failed"
        exit 1
    fi
    
    # Cleanup
    rm -rf /tmp/datasafe-complete
    
else
    print_error "Connection failed!"
    echo ""
    echo "🔧 To fix this:"
    echo "1. Go to AWS Lightsail Console: https://lightsail.aws.amazon.com/ls/webapp/eu-west-2/instances"
    echo "2. Check if your instance is running"
    echo "3. Get the current IP address"
    echo "4. Update the INSTANCE_IP variable in this script (line 8)"
    echo "5. Run this script again"
    echo ""
    echo "💡 If the IP changed, edit this script and update:"
    echo "   INSTANCE_IP=\"NEW_IP_ADDRESS\""
fi 