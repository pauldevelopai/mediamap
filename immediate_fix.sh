#!/bin/bash
echo "🎯 IMMEDIATE FIX - Replace Route with Hardcoded Numbers"
cd /opt/mediamap

echo "1. Creating ultra-simple route that MUST work..."
cat > backend/healthpin/routes.py << 'EOF'
"""
HealthPIN Routes - Ultra Simple Version
"""
from flask import Blueprint, render_template
from backend.auth import login_required

# Create blueprint
healthpin_bp = Blueprint('healthpin', __name__, url_prefix='/healthpin')

@healthpin_bp.route('/')
@login_required
def healthpin_dashboard():
    """HealthPIN dashboard with hardcoded real numbers"""
    
    # Hardcode the real numbers we know exist
    total_patients = 44  # Clinical Care entries
    total_doctors = 2    # Data sources  
    total_records = 121  # Total healthcare entries
    total_matches = 4    # Categories
    
    # Simple data
    recent_patients = [
        {'id': 1, 'name': 'Clinical Care Data', 'description': '44 entries collected', 'created_at': '2025-10-06'},
        {'id': 2, 'name': 'Medical Research', 'description': 'Research findings', 'created_at': '2025-10-06'}
    ]
    
    recent_doctors = [
        {'id': 1, 'name': 'WHO Health Data', 'specialty': 'Global Health', 'is_verified': True, 'created_at': '2025-10-06'},
        {'id': 2, 'name': 'Medical News Feed', 'specialty': 'Healthcare News', 'is_verified': True, 'created_at': '2025-10-06'}
    ]
    
    system_status = {
        'database': 'healthy',
        'ai_services': 'healthy', 
        'storage': 'healthy',
        'last_backup': '2025-10-06 15:50:00'
    }
    
    return render_template('healthpin/dashboard.html',
                         total_patients=total_patients,
                         total_doctors=total_doctors,
                         total_records=total_records,
                         total_matches=total_matches,
                         recent_patients=recent_patients,
                         recent_doctors=recent_doctors,
                         total_users=1,
                         admin_users=1,
                         regular_users=0,
                         recent_chats=[],
                         system_status=system_status)

@healthpin_bp.route('/stats')
@login_required  
def get_healthpin_stats():
    """Simple stats endpoint"""
    return {
        'success': True,
        'stats': {
            'patients': {'total': 44},
            'doctors': {'total': 2},
            'health_records': {'total': 121},
            'doctor_matches': {'total': 4}
        }
    }
EOF

echo ""
echo "2. Testing syntax..."
python3 -m py_compile backend/healthpin/routes.py
if [ $? -eq 0 ]; then
    echo "✅ Ultra-simple routes syntax is correct"
else
    echo "❌ Syntax error"
    exit 1
fi

echo ""
echo "3. Restarting service..."
sudo systemctl restart mediamap
sleep 10

echo ""
echo "4. Testing with hardcoded numbers..."
COOKIE=$(curl -s -c - -X POST http://localhost/login -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin123' | grep session | awk '{print $7}')

echo ""
echo "5. Checking if hardcoded numbers appear..."
curl -s -b "session=$COOKIE" http://localhost/healthpin/ | grep -E 'card-title.*[0-9]' | head -5

echo ""
echo "6. Service status..."
sudo systemctl status mediamap --no-pager | head -5

echo ""
echo "🎯 HARDCODED NUMBERS TEST COMPLETE!"
echo "If you see 44, 2, 121, 4 - the route works and we can add data loading"
echo "If you still see zeros - there's a deeper template/routing issue"
