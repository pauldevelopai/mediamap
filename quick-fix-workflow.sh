#!/bin/bash
# Quick fix workflow for Lightsail deployment

LIGHTSAIL_IP="35.176.169.218"
KEY_FILE="LightsailDefaultKey-eu-west-2.pem"

echo "🔧 Quick Fix Workflow for Lightsail"
echo "=================================="

# 1. Make your changes locally first
echo "1. Make your changes locally and test them"
echo "2. Commit changes to git:"
echo "   git add ."
echo "   git commit -m 'Fix: description of bug fix'"
echo "   git push origin main"
echo ""

# 2. Deploy to Lightsail
echo "3. Deploying to Lightsail..."
ssh -i "$KEY_FILE" ubuntu@"$LIGHTSAIL_IP" << 'EOF'
cd /opt/mediamap
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
pkill -f "python -m flask"
sleep 2
export FLASK_APP=backend/app.py
export FLASK_ENV=production
nohup python -m flask run --host=0.0.0.0 --port=8000 > app.log 2>&1 &
echo "✅ Application restarted with latest changes"
EOF

echo ""
echo "🌐 Test your fix at: http://$LIGHTSAIL_IP:8000"
echo "📋 Check logs with: ssh -i $KEY_FILE ubuntu@$LIGHTSAIL_IP 'cd /opt/mediamap && tail -f app.log'"
