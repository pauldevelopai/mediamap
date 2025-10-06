#!/bin/bash
echo "🚨 EMERGENCY RESTORE - Get Back to Working Zeros"
cd /opt/mediamap

echo "1. Restoring original routes from git..."
git checkout HEAD -- backend/healthpin/routes.py
echo "✅ Restored original routes"

echo ""
echo "2. Restarting service..."
sudo systemctl restart mediamap
sleep 10

echo ""
echo "3. Testing login page..."
curl -s http://localhost/login | head -2

echo ""
echo "4. Testing HealthPIN page (should work with zeros)..."
COOKIE=$(curl -s -c - -X POST http://localhost/login -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin123' | grep session | awk '{print $7}')

echo ""
echo "5. Checking HealthPIN response..."
curl -s -b "session=$COOKIE" http://localhost/healthpin/ | head -10

echo ""
echo "6. Testing external access..."
curl -I http://35.177.61.112/healthpin/ 2>/dev/null | head -2

echo ""
echo "7. Service status..."
sudo systemctl status mediamap --no-pager | head -5

echo ""
echo "🚨 EMERGENCY RESTORE COMPLETE!"
echo "HealthPIN should be working again (with zeros)"
echo "Visit: http://35.177.61.112/healthpin/"
