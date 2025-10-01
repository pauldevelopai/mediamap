#!/bin/bash
# Debug workflow for Lightsail deployment

LIGHTSAIL_IP="35.176.169.218"
KEY_FILE="LightsailDefaultKey-eu-west-2.pem"

echo "🐛 Debug Workflow for Lightsail"
echo "=============================="

# 1. Check application status
echo "1. Checking application status..."
ssh -i "$KEY_FILE" ubuntu@"$LIGHTSAIL_IP" << 'EOF'
cd /opt/mediamap
echo "📊 Process Status:"
ps aux | grep flask
echo ""
echo "📋 Recent Logs:"
tail -20 app.log
echo ""
echo "💾 Disk Space:"
df -h
echo ""
echo "🔍 Port Status:"
netstat -tlnp | grep :8000
EOF

echo ""
echo "2. Enable debug mode for detailed error tracking..."
ssh -i "$KEY_FILE" ubuntu@"$LIGHTSAIL_IP" << 'EOF'
cd /opt/mediamap
pkill -f "python -m flask"
sleep 2
source venv/bin/activate
export FLASK_APP=backend/app.py
export FLASK_ENV=development
export FLASK_DEBUG=1
nohup python -m flask run --host=0.0.0.0 --port=8000 > app_debug.log 2>&1 &
echo "✅ Debug mode enabled"
EOF

echo ""
echo "3. Monitor logs in real-time:"
echo "   ssh -i $KEY_FILE ubuntu@$LIGHTSAIL_IP 'cd /opt/mediamap && tail -f app_debug.log'"
echo ""
echo "4. Test specific endpoints:"
echo "   curl http://$LIGHTSAIL_IP:8000/health"
echo "   curl -X POST http://$LIGHTSAIL_IP:8000/login -d 'section=admin&username=admin&password=admin123'"
