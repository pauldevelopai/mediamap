#!/bin/bash
echo "🔧 Restarting HealthPIN service with updated dashboard"
cd /opt/mediamap
sudo systemctl restart mediamap
sleep 8
echo "✅ Service restarted"
echo ""
echo "Testing HealthPIN dashboard:"
curl -s http://localhost/login | head -2
echo ""
echo "🎯 HealthPIN Dashboard should now show real agent data!"
echo "Visit: http://35.177.61.112/healthpin/"
