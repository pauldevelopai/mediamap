#!/bin/bash
echo "🏥 Testing Doctor Scraping on Lightsail"
echo "======================================"

# Test if the endpoint is accessible
echo "1. Testing HealthPIN scrape-doctors endpoint..."
ssh -i LightsailDefaultKey-eu-west-2.pem -o StrictHostKeyChecking=no ubuntu@35.177.61.112 "
cd /opt/mediamap && source venv/bin/activate

echo '🔍 Checking if HealthPIN agent has doctor scraping method...'
python3 -c \"
import sys
sys.path.append('/opt/mediamap/backend')

try:
    from backend.agents.agent_manager import agent_manager
    
    print(f'Available agents: {list(agent_manager.agents.keys())}')
    
    if 'healthpin' in agent_manager.agents:
        agent = agent_manager.agents['healthpin']
        print(f'HealthPIN agent type: {type(agent)}')
        
        if hasattr(agent, 'scrape_doctors_south_africa'):
            print('✅ scrape_doctors_south_africa method exists')
        else:
            print('❌ scrape_doctors_south_africa method NOT found')
            print('Available methods:')
            methods = [m for m in dir(agent) if not m.startswith('_')]
            for method in methods[:10]:
                print(f'  - {method}')
    else:
        print('❌ HealthPIN agent not found in agent_manager')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
\"

echo ''
echo '📊 Checking current doctor count in database...'
python3 -c \"
import sys
sys.path.append('/opt/mediamap/backend')

try:
    from backend.app import app
    
    with app.app_context():
        from backend.healthpin.models import Doctor
        count = Doctor.query.count()
        print(f'Current doctors in database: {count}')
        
        if count > 0:
            doctors = Doctor.query.limit(3).all()
            print('Sample doctors:')
            for doc in doctors:
                print(f'  - {doc.name} in {doc.city}, {doc.province}')
        
except Exception as e:
    print(f'Error checking database: {e}')
\"

echo ''
echo '🔧 Checking service logs for any errors...'
sudo journalctl -u mediamap -n 10 --no-pager | grep -i error || echo 'No recent errors found'
"

echo ""
echo "2. Manual trigger test (if you want to run it):"
echo "   Visit: http://35.177.61.112/healthpin/doctors"
echo "   Click the 'Scrape More Doctors' button"
echo ""
echo "3. Or trigger via curl (requires login session):"
echo "   curl -X POST http://35.177.61.112/healthpin/scrape-doctors \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"limit\": 20}' \\"
echo "        -b 'session=your_session_cookie'"
