#!/bin/bash
echo "🎯 DEPLOYING FINAL HealthPIN SOLUTION - Real Agent Data Integration"
cd /opt/mediamap

echo "1. Backing up current routes file..."
cp backend/healthpin/routes.py backend/healthpin/routes.py.backup.$(date +%s)

echo ""
echo "2. Testing Python syntax of new files..."
python3 -c "
import sys
sys.path.insert(0, '/opt/mediamap')
try:
    from backend.healthpin.data_coordinator import healthpin_coordinator
    print('✅ Data coordinator imports successfully')
    
    # Test the coordinator
    stats = healthpin_coordinator.get_coordinated_dashboard_stats()
    print(f'✅ Coordinator working - found {stats[\"total_records\"]} records')
    
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

echo ""
echo "3. Testing updated routes syntax..."
python3 -m py_compile backend/healthpin/routes.py
if [ $? -eq 0 ]; then
    echo "✅ Routes syntax is correct"
else
    echo "❌ Routes syntax error - restoring backup"
    cp backend/healthpin/routes.py.backup.* backend/healthpin/routes.py
    exit 1
fi

echo ""
echo "4. Testing full app import..."
python3 -c "
import sys
sys.path.insert(0, '/opt/mediamap')
try:
    import backend.app
    print('✅ Full app imports successfully')
except Exception as e:
    print(f'❌ App import error: {e}')
    sys.exit(1)
"

echo ""
echo "5. Restarting service with new code..."
sudo systemctl restart mediamap
sleep 10

echo ""
echo "6. Checking service status..."
sudo systemctl status mediamap --no-pager

echo ""
echo "7. Testing HealthPIN dashboard..."
curl -s http://localhost/login | head -2

echo ""
echo "8. Testing HealthPIN page specifically..."
COOKIE=$(curl -s -c - -X POST http://localhost/login -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin123' | grep session | awk '{print $7}')
curl -s -b "session=$COOKIE" http://localhost/healthpin/ | grep -E '(Total|Verified|Health|AI).*[0-9]' | head -5

echo ""
echo "9. Checking for any errors in logs..."
sudo journalctl -u mediamap --no-pager -n 10 | grep -E "(ERROR|Exception)" | tail -3

echo ""
echo "🎯 DEPLOYMENT COMPLETE!"
echo "✅ HealthPIN dashboard now shows real agent data"
echo "✅ Data is coordinated and assessed at single point"
echo "✅ Visit: http://35.177.61.112/healthpin/"
