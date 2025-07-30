#!/bin/bash

# Quick Lightsail Update Script
# Update this IP address when your instance is running

echo "🚀 Quick MediaMap Lightsail Update"
echo "=================================="

# UPDATE THIS IP ADDRESS when your instance is running
INSTANCE_IP="35.176.169.218"
SSH_KEY="LightsailDefaultKey-eu-west-2.pem"

echo "📍 Current IP: $INSTANCE_IP"
echo "🔑 SSH Key: $SSH_KEY"
echo ""

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    echo "💡 Please download your Lightsail SSH key first"
    exit 1
fi

# Test connection
echo "🔍 Testing connection to $INSTANCE_IP..."
if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "echo 'Connection successful'" 2>/dev/null; then
    echo "✅ Connection successful! Deploying updates..."
    echo ""
    
    # Create temp directory and copy files
    mkdir -p /tmp/mediamap-quick
    cp backend/templates/user_dashboard.html /tmp/mediamap-quick/
    cp backend/templates/user_chats.html /tmp/mediamap-quick/
    cp backend/templates/login.html /tmp/mediamap-quick/
    
    # Upload and deploy
    echo "📤 Uploading files..."
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r /tmp/mediamap-quick/* ubuntu@$INSTANCE_IP:/tmp/
    
    echo "🔧 Deploying on server..."
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
        
        echo "⏳ Waiting for restart..."
        sleep 10
        
        echo "✅ Update complete!"
EOF
    
    echo ""
    echo "🎉 Deployment successful!"
    echo "🌐 Your updated app is now live at: http://$INSTANCE_IP"
    echo ""
    echo "✨ Updated features:"
    echo "   • Username display in top bar"
    echo "   • Working chat functionality"
    echo "   • Conversation saving indicator"
    echo "   • Feedback system on my-chats page"
    echo "   • Modern login page"
    
    # Cleanup
    rm -rf /tmp/mediamap-quick
    
else
    echo "❌ Connection failed!"
    echo ""
    echo "🔧 To fix this:"
    echo "1. Go to AWS Lightsail Console: https://lightsail.aws.amazon.com/ls/webapp/eu-west-2/instances"
    echo "2. Check if your instance is running"
    echo "3. Get the current IP address"
    echo "4. Update the INSTANCE_IP variable in this script"
    echo "5. Run this script again"
    echo ""
    echo "💡 If the IP changed, edit this script and update line 8:"
    echo "   INSTANCE_IP=\"NEW_IP_ADDRESS\""
fi 