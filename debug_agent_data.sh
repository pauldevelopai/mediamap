#!/bin/bash
echo "🔍 DEBUGGING: Why HealthPIN still shows zeros"
cd /opt/mediamap

echo "1. Checking if agent data file exists..."
ls -la backend/agents/storage/healthpin/HealthPINAgent_data.json
echo ""

echo "2. Checking file contents and size..."
if [ -f "backend/agents/storage/healthpin/HealthPINAgent_data.json" ]; then
    echo "File size: $(wc -c < backend/agents/storage/healthpin/HealthPINAgent_data.json) bytes"
    echo "First few lines:"
    head -5 backend/agents/storage/healthpin/HealthPINAgent_data.json
    echo ""
    echo "Entry count:"
    python3 -c "
import json
try:
    with open('backend/agents/storage/healthpin/HealthPINAgent_data.json', 'r') as f:
        data = json.load(f)
    print(f'Total entries: {len(data)}')
    if data:
        print(f'Sample entry: {data[0]}')
except Exception as e:
    print(f'Error reading file: {e}')
"
else
    echo "❌ Agent data file does not exist!"
fi

echo ""
echo "3. Testing data coordinator directly..."
python3 -c "
import sys
sys.path.insert(0, '/opt/mediamap')
try:
    from backend.healthpin.data_coordinator import healthpin_coordinator
    stats = healthpin_coordinator.get_coordinated_dashboard_stats()
    print('Data coordinator results:')
    print(f'  Total patients: {stats[\"total_patients\"]}')
    print(f'  Total doctors: {stats[\"total_doctors\"]}')
    print(f'  Total records: {stats[\"total_records\"]}')
    print(f'  Total matches: {stats[\"total_matches\"]}')
    print(f'  Collection status: {stats[\"collection_status\"]}')
    print(f'  Data file path being used: {healthpin_coordinator.agent_data_file}')
except Exception as e:
    print(f'❌ Data coordinator error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "4. Checking all possible agent data locations..."
find /opt/mediamap -name "*HealthPIN*data*" -type f 2>/dev/null
find /opt/mediamap -name "*healthpin*" -type f | grep -i data

echo ""
echo "5. Testing HealthPIN dashboard route directly..."
python3 -c "
import sys
sys.path.insert(0, '/opt/mediamap')
try:
    from backend.healthpin.routes import healthpin_bp
    print('✅ HealthPIN routes import successfully')
except Exception as e:
    print(f'❌ Routes import error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "6. Checking recent service logs for errors..."
sudo journalctl -u mediamap --no-pager -n 20 | grep -E "(coordinator|HealthPIN|agent_data)"

echo ""
echo "🎯 DIAGNOSIS COMPLETE - Check results above!"
