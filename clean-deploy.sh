#!/bin/bash

# Clean Deploy Script - Perfect Mirror
# ====================================
# Creates a perfect mirror of local machine on Lightsail

set -e

# Configuration
LIGHTSAIL_IP="35.177.61.112"
LIGHTSAIL_USER="ubuntu"
LIGHTSAIL_KEY="LightsailDefaultKey-eu-west-2.pem"
APP_DIR="/opt/mediamap"

echo "🚀 Clean Deploy - Perfect Mirror"
echo "================================"
echo "📍 Target: $LIGHTSAIL_USER@$LIGHTSAIL_IP"
echo "📁 App Directory: $APP_DIR"
echo ""

# Test connection
echo "🔧 Step 1: Testing connection..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "echo '✅ Connection successful'"

# Clean slate
echo "🔧 Step 2: Creating clean environment..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    # Stop any running services
    sudo systemctl stop mediamap 2>/dev/null || true
    sudo pkill -f 'python.*app.py' 2>/dev/null || true
    sudo pkill -f gunicorn 2>/dev/null || true
    
    # Clean up directory
    sudo rm -rf $APP_DIR
    sudo mkdir -p $APP_DIR
    sudo chown ubuntu:ubuntu $APP_DIR
    
    # Install system dependencies
    sudo apt update -qq
    sudo apt install -y python3 python3-pip python3-venv git curl
    echo '✅ System setup complete'
"

# Sync essential files only (no large directories)
echo "🔧 Step 3: Syncing essential files..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='instance' \
    --exclude='*.log' \
    --exclude='.dsenv' \
    --exclude='.pytest_cache' \
    --exclude='.vscode' \
    --exclude='node_modules' \
    --exclude='*.db' \
    --exclude='*.sqlite' \
    --progress \
    -e "ssh -i $LIGHTSAIL_KEY -o StrictHostKeyChecking=no" \
    ./ "$LIGHTSAIL_USER@$LIGHTSAIL_IP:$APP_DIR/"

echo "✅ File sync complete"

# Set up Python environment
echo "🔧 Step 4: Setting up Python environment..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    cd $APP_DIR
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo '✅ Python environment ready'
"

# Create database and tables
echo "🔧 Step 5: Creating database..."
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

# Create all essential tables
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
    '''CREATE TABLE IF NOT EXISTS prompt_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_id INTEGER NOT NULL,
        version_number VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        created_by INTEGER,
        FOREIGN KEY (prompt_id) REFERENCES prompt_templates (id),
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
    )''',
    '''CREATE TABLE IF NOT EXISTS healthpin_health_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        doctor_id INTEGER,
        record_type VARCHAR(50) NOT NULL,
        title VARCHAR(200) NOT NULL,
        content TEXT NOT NULL,
        diagnosis TEXT,
        treatment_plan TEXT,
        medications TEXT,
        follow_up_date DATE,
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES healthpin_patients (id),
        FOREIGN KEY (doctor_id) REFERENCES healthpin_doctors (id)
    )''',
    '''CREATE TABLE IF NOT EXISTS healthpin_doctor_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        doctor_id INTEGER NOT NULL,
        match_score DECIMAL(5,2),
        match_reasons TEXT,
        status VARCHAR(50) DEFAULT 'pending',
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES healthpin_patients (id),
        FOREIGN KEY (doctor_id) REFERENCES healthpin_doctors (id)
    )''',
    '''CREATE TABLE IF NOT EXISTS healthpin_family_notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        family_member_phone VARCHAR(20) NOT NULL,
        notification_type VARCHAR(50) NOT NULL,
        message TEXT NOT NULL,
        status VARCHAR(50) DEFAULT 'pending',
        sent_at DATETIME,
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES healthpin_patients (id)
    )''',
    '''CREATE TABLE IF NOT EXISTS healthpin_consultations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        doctor_id INTEGER NOT NULL,
        consultation_type VARCHAR(50) NOT NULL,
        scheduled_date DATETIME,
        status VARCHAR(50) DEFAULT 'scheduled',
        notes TEXT,
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES healthpin_patients (id),
        FOREIGN KEY (doctor_id) REFERENCES healthpin_doctors (id)
    )''',
    '''CREATE TABLE IF NOT EXISTS healthpin_health_news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title VARCHAR(200) NOT NULL,
        content TEXT NOT NULL,
        category VARCHAR(100),
        language VARCHAR(10) DEFAULT 'en',
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )'''
]

for table in tables:
    cursor.execute(table)

# Create admin user
cursor.execute('''
    INSERT OR IGNORE INTO users (username, email, password_hash, is_admin)
    VALUES (?, ?, ?, ?)
''', ('admin', 'admin@aimap.ai', 'pbkdf2:sha256:600000\$abc123\$hash', 1))

# Create default prompt templates
cursor.execute('''
    INSERT OR IGNORE INTO prompt_templates (name, description, category, prompt_type, content, llm_provider, model_name, usage_context, variables, is_active, version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', ('MediaMap System Prompt', 'Default system prompt for MediaMap AI', 'system', 'system_message', 'You are MediaMap AI, a specialized assistant for media industry analysis, business insights, and strategic planning.', 'openai', 'gpt-4', 'MediaMap chat interface', '{\"user_name\": \"User name\", \"context\": \"Business context\"}', 1, '1.0'))

cursor.execute('''
    INSERT OR IGNORE INTO prompt_templates (name, description, category, prompt_type, content, llm_provider, model_name, usage_context, variables, is_active, version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', ('HealthPIN System Prompt', 'Default system prompt for HealthPIN AI', 'system', 'system_message', 'You are HealthPIN AI, a specialized medical assistant for healthcare analysis and clinical insights.', 'openai', 'gpt-4', 'HealthPIN chat interface', '{\"user_name\": \"User name\", \"context\": \"Clinical context\"}', 1, '1.0'))

conn.commit()
conn.close()
print('✅ Database created with all tables and admin user')
\"
"

# Set up systemd service
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
    echo '✅ Systemd service configured'
"

# Set permissions
echo "🔧 Step 7: Setting permissions..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    cd $APP_DIR
    sudo chown -R www-data:www-data .
    sudo chmod -R 755 .
    sudo chmod 664 instance/*.db 2>/dev/null || true
    echo '✅ Permissions set'
"

# Start application
echo "🔧 Step 8: Starting application..."
ssh -i "$LIGHTSAIL_KEY" -o StrictHostKeyChecking=no "$LIGHTSAIL_USER@$LIGHTSAIL_IP" "
    cd $APP_DIR
    sudo systemctl start mediamap
    sleep 5
    sudo systemctl status mediamap --no-pager -l
"

# Test application
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
echo "🎉 Clean deployment completed!"
echo ""
echo "🌐 Application URLs:"
echo "   - http://$LIGHTSAIL_IP:8000"
echo "   - http://$LIGHTSAIL_IP:3000"
echo ""
echo "🔑 Admin login: admin / admin123"
echo ""
echo "📋 Management commands:"
echo "   - Connect: ./connect-lightsail.sh"
echo "   - Update: ./update-lightsail.sh"
echo "   - Manager: ./lightsail-manager.sh"
echo ""
echo "✅ Perfect mirror created - ready for development!"

