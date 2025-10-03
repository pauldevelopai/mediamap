#!/bin/bash

# Push Lightsail Changes to GitHub
# ================================
# This script pushes changes from Lightsail back to GitHub

set -e

# Configuration
LIGHTSAIL_IP="35.177.61.112"
LIGHTSAIL_USER="ubuntu"
LIGHTSAIL_KEY="LightsailDefaultKey-eu-west-2.pem"
APP_DIR="/opt/mediamap"

echo "🚀 Push Lightsail Changes to GitHub"
echo "=================================="
echo "📍 Target: $LIGHTSAIL_USER@$LIGHTSAIL_IP"
echo "📁 App Directory: $APP_DIR"
echo ""

# Check if key file exists
if [ ! -f "$LIGHTSAIL_KEY" ]; then
    echo "❌ SSH key file not found: $LIGHTSAIL_KEY"
    exit 1
fi

# Set proper permissions on key file
chmod 400 "$LIGHTSAIL_KEY"

# Test connection
echo "🔧 Testing connection..."
if ! ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "echo 'Connection test successful'" 2>/dev/null; then
    echo "❌ Cannot connect to Lightsail instance"
    echo "💡 Try running: ./connect-lightsail.sh"
    exit 1
fi
echo "✅ Connection test successful"

# Check git status on Lightsail
echo "🔧 Checking git status on Lightsail..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    cd $APP_DIR
    echo 'Current git status:'
    git status --porcelain
    echo ''
    echo 'Current branch:'
    git branch --show-current
"

# Ask for confirmation
echo ""
read -p "🤔 Do you want to commit and push these changes to GitHub? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Operation cancelled"
    exit 1
fi

# Commit and push changes
echo "🔧 Committing and pushing changes to GitHub..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    cd $APP_DIR
    
    # Add all changes
    git add .
    
    # Commit with timestamp
    COMMIT_MSG=\"Update from Lightsail - \$(date '+%Y-%m-%d %H:%M:%S')\"
    git commit -m \"\$COMMIT_MSG\" || echo 'No changes to commit'
    
    # Push to GitHub
    git push origin main
    
    echo '✅ Changes pushed to GitHub successfully'
"

echo ""
echo "🎉 GitHub push completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Run: ./pull-from-github.sh (to pull changes to local)"
echo "   2. Check GitHub repository for the latest changes"
