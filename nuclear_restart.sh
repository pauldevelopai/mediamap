#!/bin/bash
echo "🚨 NUCLEAR RESTART - Complete Service Recovery"
cd /opt/mediamap

echo "1. Killing ALL Python processes..."
sudo pkill -9 -f python
sudo pkill -9 -f gunicorn
sleep 3

echo ""
echo "2. Stopping service..."
sudo systemctl stop mediamap
sleep 3

echo ""
echo "3. Checking for any remaining processes..."
ps aux | grep -E "(python|gunicorn)" | grep -v grep

echo ""
echo "4. Testing app.py syntax..."
python3 -c "
import sys
sys.path.insert(0, '/opt/mediamap')
try:
    import backend.app
    print('✅ app.py imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo ""
echo "5. Checking file permissions..."
ls -la backend/app.py
sudo chown www-data:www-data backend/app.py

echo ""
echo "6. Starting service fresh..."
sudo systemctl start mediamap
sleep 15

echo ""
echo "7. Checking service status..."
sudo systemctl status mediamap --no-pager

echo ""
echo "8. Checking if gunicorn is running..."
ps aux | grep gunicorn | grep -v grep

echo ""
echo "9. Testing connection..."
curl -s http://localhost:8000/ | head -2
curl -s http://localhost/login | head -2

echo ""
echo "10. If still failing, checking recent logs..."
sudo journalctl -u mediamap --no-pager -n 10

echo ""
echo "🎯 NUCLEAR RESTART COMPLETE!"
