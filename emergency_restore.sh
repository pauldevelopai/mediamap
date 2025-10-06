#!/bin/bash
echo "🚨 EMERGENCY RESTORE - Getting Back to Working State"
cd /opt/mediamap

echo "1. Restoring from git to get back to working state..."
git checkout HEAD -- backend/healthpin/routes.py
echo "✅ Restored routes from git"

echo ""
echo "2. Restarting service..."
sudo systemctl restart mediamap
sleep 10

echo ""
echo "3. Testing if login page works..."
curl -s http://localhost/login | head -2

echo ""
echo "4. Testing if HealthPIN page loads (even with zeros)..."
COOKIE=$(curl -s -c - -X POST http://localhost/login -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin123' | grep session | awk '{print $7}')
curl -s -b "session=$COOKIE" http://localhost/healthpin/ | head -5

echo ""
echo "5. Service status..."
sudo systemctl status mediamap --no-pager | head -5

echo ""
echo "🚨 EMERGENCY RESTORE COMPLETE!"
echo "This should get you back to the working HealthPIN page (with zeros)"
echo "Then we can figure out why the data isn't showing"
