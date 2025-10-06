#!/bin/bash
echo "🎯 FORCE NUMBERS TEST - Hardcode Real Numbers"
cd /opt/mediamap

echo "1. Creating a test route with hardcoded numbers..."
cat > /tmp/test_route.py << 'EOF'
@healthpin_bp.route('/test')
@login_required
def test_dashboard():
    """Test route with hardcoded numbers"""
    return render_template('healthpin/dashboard.html',
                         total_patients=40,
                         total_doctors=2,
                         total_records=110,
                         total_matches=4,
                         recent_patients=[],
                         recent_doctors=[],
                         total_users=1,
                         admin_users=1,
                         regular_users=0,
                         recent_chats=[],
                         system_status={'database': 'healthy', 'ai_services': 'healthy', 'storage': 'healthy'})
EOF

echo ""
echo "2. Adding test route to routes file..."
cat /tmp/test_route.py >> backend/healthpin/routes.py

echo ""
echo "3. Restarting service..."
sudo systemctl restart mediamap
sleep 8

echo ""
echo "4. Testing hardcoded numbers route..."
COOKIE=$(curl -s -c - -X POST http://localhost/login -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin123' | grep session | awk '{print $7}')

echo ""
echo "5. Accessing test route..."
curl -s -b "session=$COOKIE" http://localhost/healthpin/test | grep -E 'Total.*[0-9]|card-title.*[0-9]' | head -5

echo ""
echo "6. If test route shows real numbers, the issue is with data processing"
echo "   If test route still shows zeros, the issue is with template rendering"

echo ""
echo "🎯 FORCE TEST COMPLETE!"
