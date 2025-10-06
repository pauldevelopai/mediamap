#!/bin/bash
echo "🔧 FIXING SYNTAX ERROR in app.py"
cd /opt/mediamap

echo "1. Finding the syntax error..."
grep -n "fError loading HealthPIN data" backend/app.py

echo ""
echo "2. Fixing the syntax error..."
# Fix the f-string syntax error
sed -i 's/print(fError loading HealthPIN data:/print(f"Error loading HealthPIN data:/g' backend/app.py

echo ""
echo "3. Checking if there are other similar errors..."
grep -n "print(f[^\"']" backend/app.py

echo ""
echo "4. Testing Python syntax..."
python3 -m py_compile backend/app.py
if [ $? -eq 0 ]; then
    echo "✅ Syntax is now correct!"
else
    echo "❌ Still has syntax errors"
    exit 1
fi

echo ""
echo "5. Restarting service..."
sudo systemctl restart mediamap
sleep 8

echo ""
echo "6. Testing if service is working..."
curl -s http://localhost/login | head -2

echo ""
echo "🎯 SYNTAX ERROR FIXED!"
