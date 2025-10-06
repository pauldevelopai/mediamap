#!/bin/bash
echo "🔍 DIRECT DEBUGGING - What's Actually Happening"
cd /opt/mediamap

echo "1. Checking if clean routes file was applied..."
head -10 backend/healthpin/routes.py
echo ""
echo "File size and modification time:"
ls -la backend/healthpin/routes.py

echo ""
echo "2. Testing if the route can access the data file..."
python3 -c "
import json
import os

data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
print(f'Data file exists: {os.path.exists(data_file)}')

if os.path.exists(data_file):
    with open(data_file, 'r') as f:
        agent_data = json.load(f)
    
    categories = {}
    sources = set()
    
    for entry in agent_data:
        cat = entry.get('category', 'Unknown')
        source = entry.get('source', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
        sources.add(source)
    
    total_patients = categories.get('Clinical_Care', 0)
    total_doctors = len(sources)
    total_records = len(agent_data)
    total_matches = len(categories)
    
    print(f'EXPECTED NUMBERS:')
    print(f'  Total Patients: {total_patients}')
    print(f'  Total Doctors: {total_doctors}')
    print(f'  Total Records: {total_records}')
    print(f'  Total Matches: {total_matches}')
else:
    print('❌ Data file not found!')
"

echo ""
echo "3. Checking current service status..."
sudo systemctl status mediamap --no-pager -l | head -10

echo ""
echo "4. Testing the actual HealthPIN route..."
COOKIE=$(curl -s -c - -X POST http://localhost/login -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin123' | grep session | awk '{print $7}')

echo ""
echo "5. Getting the actual HTML response..."
curl -s -b "session=$COOKIE" http://localhost/healthpin/ > /tmp/healthpin_actual.html
echo "Response saved to /tmp/healthpin_actual.html"

echo ""
echo "6. Looking for the numbers in the HTML..."
grep -E 'Total.*[0-9]|card-title.*[0-9]' /tmp/healthpin_actual.html

echo ""
echo "7. Checking for any template errors..."
grep -E 'total_patients|total_doctors|total_records|total_matches' /tmp/healthpin_actual.html | head -5

echo ""
echo "8. Recent service logs..."
sudo journalctl -u mediamap --no-pager -n 10 | tail -10

echo ""
echo "9. Testing if we can manually call the route function..."
python3 -c "
import sys
sys.path.insert(0, '/opt/mediamap')

try:
    # Test the route function directly
    import os
    os.chdir('/opt/mediamap')
    
    # Import the route
    from backend.healthpin.routes import healthpin_dashboard
    print('✅ Route imported successfully')
    
    # This won't work without Flask context, but we can see if import works
except Exception as e:
    print(f'❌ Route import error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🎯 DIRECT DEBUG COMPLETE!"
