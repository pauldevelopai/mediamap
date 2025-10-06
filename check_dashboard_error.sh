#!/bin/bash
echo "🔍 CHECKING CURRENT DASHBOARD ERROR"
cd /opt/mediamap

echo "1. Testing HealthPIN dashboard and checking for errors..."
COOKIE=$(curl -s -c - -X POST http://localhost/login -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin123' | grep session | awk '{print $7}')

echo ""
echo "2. Accessing HealthPIN dashboard..."
curl -s -b "session=$COOKIE" http://localhost/healthpin/ > /tmp/healthpin_response.html

echo ""
echo "3. Checking recent error logs..."
sudo journalctl -u mediamap --no-pager -n 20 | grep -E "(ERROR|Exception|Traceback)" | tail -10

echo ""
echo "4. Testing data coordinator directly in Flask context..."
python3 -c "
import sys
sys.path.insert(0, '/opt/mediamap')

# Test without Flask context first
try:
    from backend.healthpin.data_coordinator import healthpin_coordinator
    stats = healthpin_coordinator.get_coordinated_dashboard_stats()
    print('✅ Data coordinator works outside Flask:')
    print(f'  Records: {stats[\"total_records\"]}')
    print(f'  Patients: {stats[\"total_patients\"]}')
except Exception as e:
    print(f'❌ Data coordinator error: {e}')

# Now test with Flask app context
try:
    from backend.app import app
    with app.app_context():
        from backend.healthpin.data_coordinator import healthpin_coordinator
        stats = healthpin_coordinator.get_coordinated_dashboard_stats()
        print('✅ Data coordinator works in Flask context:')
        print(f'  Records: {stats[\"total_records\"]}')
        print(f'  Patients: {stats[\"total_patients\"]}')
except Exception as e:
    print(f'❌ Flask context error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "5. Checking if the route is even being called..."
echo "Looking for HealthPIN dashboard access in logs..."
sudo journalctl -u mediamap --no-pager -n 50 | grep -i "healthpin.*dashboard" | tail -5

echo ""
echo "🎯 ERROR DIAGNOSIS COMPLETE!"
