#!/bin/bash

# Deploy Fixes to Lightsail
# =========================
# This script deploys all fixes to the Lightsail instance

set -e

echo "🚀 Deploying fixes to Lightsail..."
echo "=================================="

# Configuration
LIGHTSAIL_IP="18.175.120.201"
LIGHTSAIL_USER="ubuntu"
LIGHTSAIL_KEY="LightsailDefaultKey-eu-west-2.pem"
REMOTE_PATH="/opt/mediamap"

# Check if key file exists
if [ ! -f "$LIGHTSAIL_KEY" ]; then
    echo "❌ SSH key file not found: $LIGHTSAIL_KEY"
    exit 1
fi

# Set proper permissions on key file
chmod 400 "$LIGHTSAIL_KEY"

echo "📁 Copying fix scripts to Lightsail..."

# Copy fix scripts
scp -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no \
    fix_all_database_issues.py \
    enable_healthpin_primary.py \
    "$LIGHTSAIL_USER@$LIGHTSAIL_IP:$REMOTE_PATH/"

echo "✅ Scripts copied successfully"

echo "🔧 Running database fixes..."

# Run the database fix script
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" \
    "cd $REMOTE_PATH && python fix_all_database_issues.py"

echo "⚙️ Configuring HealthPIN as primary agent..."

# Run the HealthPIN configuration script
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" \
    "cd $REMOTE_PATH && python enable_healthpin_primary.py"

echo "🔄 Restarting MediaMap service..."

# Restart the service
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" \
    "cd $REMOTE_PATH && sudo systemctl restart mediamap || ./start_healthpin.sh &"

echo "✅ All fixes deployed successfully!"
echo ""
echo "🎉 Summary of fixes applied:"
echo "   1. ✅ Fixed missing 'organisations' table"
echo "   2. ✅ Configured HealthPIN doctor scraping"
echo "   3. ✅ Set HealthPIN as primary agent"
echo "   4. ✅ Restarted MediaMap service"
echo ""
echo "🌐 Your MediaMap instance should now be working at:"
echo "   http://$LIGHTSAIL_IP:3000"
echo ""
echo "💡 HealthPIN features now available:"
echo "   - Doctor directory (South Africa)"
echo "   - Patient-doctor matching"
echo "   - Health record management"
echo "   - Multi-language support"
