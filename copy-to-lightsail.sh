#!/bin/bash

echo "📁 Preparing files for Lightsail deployment..."
echo "=============================================="

# Create a deployment directory
mkdir -p /tmp/lightsail-deploy

# Copy the updated template files
cp backend/templates/user_dashboard.html /tmp/lightsail-deploy/
cp backend/templates/user_chats.html /tmp/lightsail-deploy/
cp backend/templates/login.html /tmp/lightsail-deploy/

# Create a simple deployment script
cat > /tmp/lightsail-deploy/deploy.sh << 'EOF'
#!/bin/bash
echo "🚀 Deploying DataSafe updates..."

# Navigate to app directory
cd /opt/datasafe

# Create backup
sudo cp -r backend/templates backend/templates.backup.$(date +%Y%m%d_%H%M%S)

# Copy new templates
sudo cp /tmp/lightsail-deploy/*.html backend/templates/

# Set permissions
sudo chown ubuntu:ubuntu backend/templates/*.html
sudo chmod 644 backend/templates/*.html

# Restart application
docker-compose restart

# Wait for restart
sleep 15

# Check status
docker-compose ps

echo "✅ Deployment complete!"
echo "🌐 Your app is now live at: http://35.176.169.218:8000"
EOF

chmod +x /tmp/lightsail-deploy/deploy.sh

echo "✅ Files prepared in /tmp/lightsail-deploy/"
echo ""
echo "📋 Next steps:"
echo "1. In your browser SSH terminal, run:"
echo "   mkdir -p /tmp/lightsail-deploy"
echo ""
echo "2. Copy the contents of these files from your local machine:"
echo "   - /tmp/lightsail-deploy/user_dashboard.html"
echo "   - /tmp/lightsail-deploy/user_chats.html"
echo "   - /tmp/lightsail-deploy/login.html"
echo "   - /tmp/lightsail-deploy/deploy.sh"
echo ""
echo "3. Run the deployment script:"
echo "   bash /tmp/lightsail-deploy/deploy.sh"
echo ""
echo "📁 Files are ready in: /tmp/lightsail-deploy/" 