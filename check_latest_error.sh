#!/bin/bash
echo "🚨 CHECKING LATEST INTERNAL SERVER ERROR"
cd /opt/mediamap

echo "1. Most recent error logs..."
sudo journalctl -u mediamap --no-pager -n 30 | grep -A 10 -B 5 "ERROR\|Exception\|Traceback" | tail -20

echo ""
echo "2. Testing if we can access any other pages..."
curl -s http://localhost/login | head -2
curl -s http://localhost/admin | head -2

echo ""
echo "3. Checking if HealthPIN blueprint is causing app-wide issues..."
python3 -c "
import sys
sys.path.insert(0, '/opt/mediamap')
try:
    # Test importing the routes file
    from backend.healthpin.routes import healthpin_bp
    print('✅ HealthPIN routes import successfully')
    print(f'Blueprint name: {healthpin_bp.name}')
    print(f'URL prefix: {healthpin_bp.url_prefix}')
except Exception as e:
    print(f'❌ HealthPIN routes import error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "4. Service status..."
sudo systemctl status mediamap --no-pager | head -10

echo ""
echo "5. Testing a simple route to see if Flask is working..."
curl -s http://localhost/ | head -2

echo ""
echo "🚨 LATEST ERROR DIAGNOSIS COMPLETE!"
