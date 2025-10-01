#!/bin/bash

# Fresh Instance Deployment Script
# ================================
# This script deploys the application to a fresh Lightsail instance

set -e

echo "🚀 Fresh Instance Deployment"
echo "==========================="
echo ""

# Get new IP address
read -p "Enter the new Lightsail IP address: " NEW_IP

if [ -z "$NEW_IP" ]; then
    echo "❌ No IP address provided"
    exit 1
fi

echo "📍 Deploying to: $NEW_IP"
echo ""

# Configuration
LIGHTSAIL_USER="ubuntu"
LIGHTSAIL_KEY="LightsailDefaultKey-eu-west-2.pem"
APP_DIR="/opt/mediamap"

# Check if key file exists
if [ ! -f "$LIGHTSAIL_KEY" ]; then
    echo "❌ SSH key file not found: $LIGHTSAIL_KEY"
    exit 1
fi

# Set proper permissions on key file
chmod 400 "$LIGHTSAIL_KEY"

echo "🔧 Step 1: Testing connection..."
# Wait for instance to be ready
echo "⏳ Waiting for instance to be ready (60 seconds)..."
sleep 60

# Test connection with retries
for i in {1..5}; do
    if ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes "$LIGHTSAIL_USER@$NEW_IP" "echo 'Connection test successful'" 2>/dev/null; then
        echo "✅ Connection successful"
        break
    else
        echo "⏳ Attempt $i/5 failed, retrying in 30 seconds..."
        sleep 30
    fi
done

echo ""
echo "🔧 Step 2: Setting up application directory..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$NEW_IP" "
    sudo mkdir -p $APP_DIR
    sudo chown ubuntu:ubuntu $APP_DIR
    cd $APP_DIR
"

echo "🔧 Step 3: Installing dependencies..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$NEW_IP" "
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv git
"

echo "🔧 Step 4: Syncing application files..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='venv' \
    --exclude='instance' \
    --exclude='*.log' \
    -e "ssh -i $LIGHTSAIL_KEY -o StrictHostKeyChecking=no" \
    ./ "$LIGHTSAIL_USER@$NEW_IP:$APP_DIR/"

echo "🔧 Step 5: Setting up Python environment..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$NEW_IP" "
    cd $APP_DIR
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
"

echo "🔧 Step 6: Creating database and tables..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$NEW_IP" "
    cd $APP_DIR
    sudo mkdir -p instance
    sudo chown www-data:www-data instance
    sudo chmod 755 instance
    
    # Create database tables
    sudo -u www-data python3 -c \"
import sqlite3
import os

# Create database
db_path = '/opt/mediamap/instance/media_analysis.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create users table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(80) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# Create other tables
tables = [
    '''CREATE TABLE IF NOT EXISTS highlander_chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id VARCHAR(100) NOT NULL,
        message TEXT NOT NULL,
        response TEXT NOT NULL,
        context TEXT,
        category VARCHAR(100),
        processed BOOLEAN DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''',
    '''CREATE TABLE IF NOT EXISTS prompt_templates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(200) NOT NULL UNIQUE,
        description TEXT,
        category VARCHAR(100) NOT NULL,
        prompt_type VARCHAR(50) NOT NULL,
        content TEXT NOT NULL,
        llm_provider VARCHAR(50) NOT NULL,
        model_name VARCHAR(100),
        usage_context VARCHAR(200),
        variables TEXT,
        is_active BOOLEAN DEFAULT 1,
        version VARCHAR(20) DEFAULT '1.0',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        created_by INTEGER,
        FOREIGN KEY (created_by) REFERENCES users (id)
    )''',
    '''CREATE TABLE IF NOT EXISTS healthpin_patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        phone_number VARCHAR(20) NOT NULL UNIQUE,
        whatsapp_id VARCHAR(50) UNIQUE,
        first_name VARCHAR(100) NOT NULL,
        last_name VARCHAR(100) NOT NULL,
        date_of_birth DATE,
        gender VARCHAR(10),
        language_preference VARCHAR(10) DEFAULT 'en',
        city VARCHAR(100),
        province VARCHAR(100),
        country VARCHAR(100) DEFAULT 'South Africa',
        preferred_specialties TEXT,
        cultural_preferences TEXT,
        accessibility_needs TEXT,
        emergency_contact_name VARCHAR(200),
        emergency_contact_phone VARCHAR(20),
        family_members TEXT,
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )'''
]

for table in tables:
    cursor.execute(table)

# Create admin user
cursor.execute('''
    INSERT OR IGNORE INTO users (username, email, password_hash, is_admin)
    VALUES (?, ?, ?, ?)
''', ('admin', 'admin@aimap.ai', 'pbkdf2:sha256:600000\$abc123\$hash', 1))

conn.commit()
conn.close()
print('✅ Database created successfully')
\"
"

echo "🔧 Step 7: Setting up systemd service..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$NEW_IP" "
    cd $APP_DIR
    sudo tee /etc/systemd/system/mediamap.service > /dev/null << 'EOF'
[Unit]
Description=MediaMap Flask Application
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
ExecStart=$APP_DIR/venv/bin/gunicorn --config gunicorn.conf.py backend.app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable mediamap
"

echo "🔧 Step 8: Setting permissions..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$NEW_IP" "
    cd $APP_DIR
    sudo chown -R www-data:www-data .
    sudo chmod -R 755 .
    sudo chmod 664 instance/*.db 2>/dev/null || true
"

echo "🔧 Step 9: Starting application..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$NEW_IP" "
    cd $APP_DIR
    sudo systemctl start mediamap
"

echo "🔧 Step 10: Testing application..."
sleep 10
if ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$NEW_IP" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" | grep -q "200"; then
    echo "✅ Application is running successfully"
else
    echo "⚠️  Application may not be running properly"
    echo "📋 Checking logs..."
    ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$NEW_IP" "sudo journalctl -u mediamap --no-pager -n 10"
fi

echo ""
echo "🎉 Fresh instance deployment completed!"
echo ""
echo "🌐 Application URL: http://$NEW_IP:8000"
echo "🔑 Admin login: admin / admin123"
echo ""
echo "📋 Next steps:"
echo "1. Test the application in your browser"
echo "2. Update all scripts with new IP: $NEW_IP"
echo "3. Run: ./update-scripts-ip.sh $NEW_IP"
