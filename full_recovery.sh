#!/bin/bash
echo "🚨 FULL RECOVERY - Fixing 502 Bad Gateway"
cd /opt/mediamap

echo "1. Checking what's running..."
ps aux | grep gunicorn | grep -v grep
ps aux | grep python | grep -v grep

echo ""
echo "2. Killing any stuck processes..."
sudo pkill -f gunicorn
sudo pkill -f "python.*app.py"
sleep 3

echo ""
echo "3. Checking service status..."
sudo systemctl status mediamap --no-pager

echo ""
echo "4. Stopping service completely..."
sudo systemctl stop mediamap
sleep 5

echo ""
echo "5. Checking for syntax errors in app.py..."
python3 -m py_compile backend/app.py
if [ $? -ne 0 ]; then
    echo "❌ Syntax error in app.py!"
    exit 1
fi

echo ""
echo "6. Starting service fresh..."
sudo systemctl start mediamap
sleep 10

echo ""
echo "7. Checking service status..."
sudo systemctl status mediamap --no-pager

echo ""
echo "8. Checking logs for errors..."
sudo journalctl -u mediamap --no-pager -n 20

echo ""
echo "9. Testing if service responds..."
curl -s http://localhost:8000/ | head -2
curl -s http://localhost/login | head -2

echo ""
echo "🎯 Recovery attempt complete!"
