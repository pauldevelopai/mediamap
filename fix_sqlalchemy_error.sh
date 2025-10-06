#!/bin/bash
echo "🔧 FIXING SQLAlchemy Context Error - Final Fix"
cd /opt/mediamap

echo "1. Testing updated routes syntax..."
python3 -m py_compile backend/healthpin/routes.py
if [ $? -eq 0 ]; then
    echo "✅ Routes syntax is correct"
else
    echo "❌ Routes syntax error"
    exit 1
fi

echo ""
echo "2. Restarting service..."
sudo systemctl restart mediamap
sleep 8

echo ""
echo "3. Testing HealthPIN dashboard..."
COOKIE=$(curl -s -c - -X POST http://localhost/login -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin123' | grep session | awk '{print $7}')

echo ""
echo "4. Checking if dashboard loads without errors..."
curl -s -b "session=$COOKIE" http://localhost/healthpin/ | grep -E 'Total.*[0-9]' | head -5

echo ""
echo "5. Checking for SQLAlchemy errors..."
sudo journalctl -u mediamap --no-pager -n 10 | grep -E "(SQLAlchemy|Flask app)" | tail -3

echo ""
echo "🎯 SHOULD BE FIXED NOW!"
echo "Visit: http://35.177.61.112/healthpin/"
