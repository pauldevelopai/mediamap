#!/bin/bash
echo "🚨 EMERGENCY RESTART - Fixing 502 Error"
cd /opt/mediamap
echo "Stopping service..."
sudo systemctl stop mediamap
sleep 3
echo "Starting service..."
sudo systemctl start mediamap
sleep 8
echo "Checking service status..."
sudo systemctl status mediamap --no-pager -l
echo ""
echo "Testing if service is responding..."
curl -s http://localhost/login | head -2
echo ""
echo "🎯 Service should be back online!"
echo "Test at: http://35.177.61.112/login"
