#!/bin/bash
echo "🚨 CHECKING INTERNAL SERVER ERROR"
cd /opt/mediamap

echo "1. Recent error logs..."
sudo journalctl -u mediamap --no-pager -n 20 | grep -E "(ERROR|Exception|Traceback)" | tail -10

echo ""
echo "2. Full recent logs..."
sudo journalctl -u mediamap --no-pager -n 15 | tail -15

echo ""
echo "3. Service status..."
sudo systemctl status mediamap --no-pager

echo ""
echo "4. Testing if app starts at all..."
curl -s http://localhost/login | head -2

echo ""
echo "5. Checking if HealthPIN blueprint is registered..."
python3 -c "
import sys
sys.path.insert(0, '/opt/mediamap')
try:
    from backend.app import app
    print('App blueprints:')
    for bp_name, bp in app.blueprints.items():
        print(f'  {bp_name}: {bp.url_prefix}')
except Exception as e:
    print(f'Error checking blueprints: {e}')
"

echo ""
echo "🚨 ERROR DIAGNOSIS COMPLETE!"
