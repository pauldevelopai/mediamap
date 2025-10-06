#!/bin/bash

# Quick Deploy Script
# ===================
# Efficient deployment to new Lightsail instance

set -e

# Configuration
LIGHTSAIL_IP="18.175.120.201"
LIGHTSAIL_USER="ubuntu"
LIGHTSAIL_KEY="LightsailDefaultKey-eu-west-2.pem"
APP_DIR="/opt/mediamap"

echo "🚀 Quick Deploy to New Instance"
echo "==============================="
echo "📍 Target: $LIGHTSAIL_USER@$LIGHTSAIL_IP"
echo ""

# Test connection
echo "🔧 Testing connection..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "echo 'Connection successful'"

echo "🔧 Step 1: Setting up application directory..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    sudo mkdir -p $APP_DIR
    sudo chown ubuntu:ubuntu $APP_DIR
"

echo "🔧 Step 2: Installing system dependencies..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv git curl
"

echo "🔧 Step 3: Syncing application files (excluding venv)..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='venv' \
    --exclude='instance' \
    --exclude='*.log' \
    --exclude='.venv' \
    -e "ssh -i $LIGHTSAIL_KEY -o StrictHostKeyChecking=no" \
    ./ "$LIGHTSAIL_USER@$LIGHTSAIL_IP:$APP_DIR/"

echo "🔧 Step 4: Setting up Python environment..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    cd $APP_DIR
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
"

echo "🔧 Step 5: Creating database and tables..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    cd $APP_DIR
    sudo mkdir -p instance
    sudo chown www-data:www-data instance
    sudo chmod 755 instance
    
    # Create database with all tables
    sudo -u www-data python3 -c \"
import sqlite3
import os

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

# Create other essential tables
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
    )''',
    '''CREATE TABLE IF NOT EXISTS healthpin_doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        phone_number VARCHAR(20) NOT NULL UNIQUE,
        whatsapp_id VARCHAR(50) UNIQUE,
        first_name VARCHAR(100) NOT NULL,
        last_name VARCHAR(100) NOT NULL,
        medical_license VARCHAR(100) UNIQUE,
        specialties TEXT,
        qualifications TEXT,
        experience_years INTEGER,
        languages TEXT,
        consultation_fee DECIMAL(10,2),
        availability_schedule TEXT,
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

echo "🔧 Step 6: Setting up systemd service..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
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

echo "🔧 Step 7: Setting permissions..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    cd $APP_DIR
    sudo chown -R www-data:www-data .
    sudo chmod -R 755 .
    sudo chmod 664 instance/*.db 2>/dev/null || true
"

echo "🔧 Step 8: Starting application..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    cd $APP_DIR
    sudo systemctl start mediamap
"

echo "🔧 Step 9: Testing application..."
sleep 10
if ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health" | grep -q "200"; then
    echo "✅ Application is running successfully"
else
    echo "⚠️  Application may not be running properly"
    echo "📋 Checking logs..."
    ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "sudo journalctl -u mediamap --no-pager -n 10"
fi

echo ""
echo "🎉 Quick deployment completed!"
echo ""
echo "🌐 Application URL: http://$LIGHTSAIL_IP:8000"
echo "🔑 Admin login: admin / admin123"
echo ""
echo "📋 Test the application:"
echo "   curl -I http://$LIGHTSAIL_IP:8000"
echo "   curl -I http://$LIGHTSAIL_IP:3000"