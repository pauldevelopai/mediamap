#!/bin/bash
echo "🔍 DIAGNOSING 502 BAD GATEWAY ERROR"
cd /opt/mediamap

echo "=== 1. SERVICE STATUS ==="
sudo systemctl status mediamap --no-pager -l

echo ""
echo "=== 2. RECENT ERROR LOGS ==="
sudo journalctl -u mediamap --no-pager -n 30 | grep -E "(ERROR|Exception|Traceback|Failed)"

echo ""
echo "=== 3. CHECKING IF GUNICORN IS RUNNING ==="
ps aux | grep gunicorn | grep -v grep

echo ""
echo "=== 4. CHECKING PORT 8000 ==="
sudo netstat -tlnp | grep :8000

echo ""
echo "=== 5. TESTING PYTHON SYNTAX ==="
cd /opt/mediamap
python3 -c "
try:
    import backend.app
    print('✅ app.py imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
"

echo ""
echo "=== 6. CHECKING CRITICAL FILES ==="
ls -la backend/app.py
ls -la gunicorn.conf.py

echo ""
echo "=== 7. TESTING MANUAL START ==="
echo "Attempting to start gunicorn manually..."
cd /opt/mediamap
timeout 10s python3 -m gunicorn --config gunicorn.conf.py backend.app:app 2>&1 | head -10

echo ""
echo "=== 8. NGINX STATUS ==="
sudo systemctl status nginx --no-pager

echo ""
echo "🎯 DIAGNOSIS COMPLETE - Check the errors above!"
