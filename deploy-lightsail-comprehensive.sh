#!/bin/bash

# Comprehensive MediaMap Lightsail Deployment Script
# This script handles multiple deployment scenarios

echo "🚀 MediaMap Lightsail Deployment Script"
echo "========================================"

# Configuration
INSTANCE_IP="13.40.124.51"
SSH_KEY="LightsailDefaultKey-eu-west-2.pem"
REGION="eu-west-2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to check if instance is reachable
check_instance() {
    print_status "Checking instance connectivity..."
    
    # Try ping
    if ping -c 1 -W 5 $INSTANCE_IP > /dev/null 2>&1; then
        print_success "Instance is reachable via ping"
        return 0
    else
        print_warning "Instance not responding to ping"
    fi
    
    # Try SSH connection
    if ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "echo 'test'" > /dev/null 2>&1; then
        print_success "SSH connection successful"
        return 0
    else
        print_error "SSH connection failed"
        return 1
    fi
}

# Function to deploy via SSH
deploy_via_ssh() {
    print_status "Deploying via SSH..."
    
    # Create temporary directory
    mkdir -p /tmp/mediamap-deploy
    
    # Copy updated files
    cp backend/templates/user_dashboard.html /tmp/mediamap-deploy/
    cp backend/templates/user_chats.html /tmp/mediamap-deploy/
    cp backend/templates/login.html /tmp/mediamap-deploy/
    
    # Upload files
    print_status "Uploading files to instance..."
    if scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r /tmp/mediamap-deploy/* ubuntu@$INSTANCE_IP:/tmp/; then
        print_success "Files uploaded successfully"
    else
        print_error "File upload failed"
        return 1
    fi
    
    # Deploy on server
    print_status "Deploying on server..."
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'EOF'
        echo "📝 Updating templates..."
        sudo cp /tmp/user_dashboard.html /opt/mediamap/backend/templates/
        sudo cp /tmp/user_chats.html /opt/mediamap/backend/templates/
        sudo cp /tmp/login.html /opt/mediamap/backend/templates/
        
        sudo chown ubuntu:ubuntu /opt/mediamap/backend/templates/*.html
        sudo chmod 644 /opt/mediamap/backend/templates/*.html
        
        echo "🔄 Restarting application..."
        cd /opt/mediamap
        docker-compose restart
        
        echo "⏳ Waiting for application to restart..."
        sleep 15
        
        echo "✅ Deployment complete!"
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Deployment completed successfully!"
        print_success "Your updated app is now live at: http://$INSTANCE_IP"
    else
        print_error "Deployment failed"
        return 1
    fi
    
    # Cleanup
    rm -rf /tmp/mediamap-deploy
}

# Function to provide manual deployment instructions
manual_deployment_guide() {
    echo ""
    print_warning "Manual Deployment Required"
    echo "================================"
    echo ""
    echo "Since the instance is not reachable, you'll need to:"
    echo ""
    echo "1. 📱 Access AWS Lightsail Console:"
    echo "   https://lightsail.aws.amazon.com/ls/webapp/eu-west-2/instances"
    echo ""
    echo "2. 🔍 Check your instance status:"
    echo "   - Is it running?"
    echo "   - Has the IP address changed?"
    echo "   - Are there any error messages?"
    echo ""
    echo "3. 🔧 If instance is stopped:"
    echo "   - Start the instance"
    echo "   - Wait for it to fully boot"
    echo "   - Get the new IP address if it changed"
    echo ""
    echo "4. 📁 Manual file upload (if needed):"
    echo "   - Use the browser-based SSH in Lightsail console"
    echo "   - Upload the updated template files"
    echo "   - Restart the application"
    echo ""
    echo "5. 🔄 Update this script with new IP (if changed):"
    echo "   Edit the INSTANCE_IP variable in this script"
    echo ""
}

# Function to test local deployment
test_local() {
    print_status "Testing local deployment..."
    
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "Local server is running"
        print_success "Test the new design at: http://localhost:8000"
        echo ""
        echo "✨ New features to test:"
        echo "   • Username display in top bar"
        echo "   • Working chat functionality"
        echo "   • Conversation saving indicator"
        echo "   • Feedback system on my-chats page"
        echo "   • Modern login page"
    else
        print_warning "Local server not running"
        echo "Start it with: cd backend && python app.py"
    fi
}

# Main deployment logic
main() {
    echo ""
    print_status "Starting deployment process..."
    echo ""
    
    # Check if SSH key exists
    if [ ! -f "$SSH_KEY" ]; then
        print_error "SSH key not found: $SSH_KEY"
        echo "Please download your Lightsail SSH key first"
        exit 1
    fi
    
    # Check instance connectivity
    if check_instance; then
        print_success "Instance is accessible, proceeding with deployment..."
        deploy_via_ssh
    else
        print_error "Instance is not accessible"
        manual_deployment_guide
        test_local
    fi
}

# Run main function
main 