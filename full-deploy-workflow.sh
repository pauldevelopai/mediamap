#!/bin/bash
# Complete deployment workflow for Lightsail

LIGHTSAIL_IP="35.176.169.218"
KEY_FILE="LightsailDefaultKey-eu-west-2.pem"

echo "🚀 Complete Deployment Workflow"
echo "==============================="

# 1. Local testing
echo "1. Testing locally first..."
if [ -f "backend/app.py" ]; then
    echo "✅ Found backend/app.py"
    echo "   Run: source venv/bin/activate && python -m flask run"
    echo "   Test your changes at http://localhost:5000"
else
    echo "❌ backend/app.py not found"
    exit 1
fi

echo ""
read -p "Press Enter after testing locally..."

# 2. Git operations
echo "2. Committing changes..."
git add .
git status
echo ""
read -p "Enter commit message: " commit_msg
git commit -m "$commit_msg"
git push origin main

# 3. Deploy to Lightsail
echo "3. Deploying to Lightsail..."
ssh -i "$KEY_FILE" ubuntu@"$LIGHTSAIL_IP" << 'EOF'
cd /opt/mediamap
echo "📥 Pulling latest changes..."
git pull origin main

echo "🔧 Updating dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo "🔄 Restarting application..."
pkill -f "python -m flask"
sleep 3

echo "🚀 Starting application..."
export FLASK_APP=backend/app.py
export FLASK_ENV=production
nohup python -m flask run --host=0.0.0.0 --port=8000 > app.log 2>&1 &

echo "⏳ Waiting for application to start..."
sleep 10

echo "🏥 Health check..."
if curl -f http://localhost:8000/health; then
    echo "✅ Application is healthy!"
else
    echo "❌ Health check failed. Checking logs..."
    tail -20 app.log
fi
EOF

echo ""
echo "🌐 Your application is available at: http://$LIGHTSAIL_IP:8000"
echo "📋 Monitor logs: ssh -i $KEY_FILE ubuntu@$LIGHTSAIL_IP 'cd /opt/mediamap && tail -f app.log'"
