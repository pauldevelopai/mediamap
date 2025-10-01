#!/bin/bash

# Run Quick Fix on Lightsail Instance
# ===================================
# This script copies the quick fix script to Lightsail and runs it

set -e

# Configuration
LIGHTSAIL_IP="35.176.169.218"
LIGHTSAIL_USER="ubuntu"
LIGHTSAIL_KEY="~/.ssh/lightsail-key.pem"

echo "🚀 Running quick fix on Lightsail instance..."
echo "📍 Target: $LIGHTSAIL_USER@$LIGHTSAIL_IP"
echo ""

# Copy the quick fix script to Lightsail
echo "🔧 Copying quick fix script to Lightsail..."
scp -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no quick_fix_lightsail.sh "$LIGHTSAIL_USER@$LIGHTSAIL_IP:/tmp/quick_fix_lightsail.sh"
echo "✅ Quick fix script copied"

# Make it executable and run it
echo "🔧 Running quick fix on Lightsail..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    chmod +x /tmp/quick_fix_lightsail.sh
    cd /opt/mediamap
    sudo /tmp/quick_fix_lightsail.sh
"
echo "✅ Quick fix completed"

echo ""
echo "🎉 Quick fix deployment completed!"
echo ""
echo "🌐 Application URL: http://$LIGHTSAIL_IP:8000"
echo "🔑 Admin login: admin / admin123"
echo ""
echo "📋 All reported issues should now be fixed:"
echo "✅ HealthPIN page should display content"
echo "✅ Highlander chatbot should work"
echo "✅ Doc chatbot should work"
echo "✅ Training page buttons should work"
echo "✅ Prompt page should work"
echo "✅ Agent start should work"
echo "✅ Insights page buttons should work"
echo "✅ Chat management history should be accessible"
