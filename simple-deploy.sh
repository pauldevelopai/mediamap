#!/bin/bash

# Simple deployment script for MediaMap
echo "🚀 Starting simple MediaMap deployment..."

# Update system
sudo apt update

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv nginx

# Create virtual environment
cd /opt/mediamap
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/mediamap.service > /dev/null <<EOF
[Unit]
Description=MediaMap Flask Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/mediamap
Environment=PATH=/opt/mediamap/venv/bin
ExecStart=/opt/mediamap/venv/bin/gunicorn --bind 0.0.0.0:8000 backend.app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Configure nginx
sudo tee /etc/nginx/sites-available/mediamap > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/mediamap /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Start services
sudo systemctl daemon-reload
sudo systemctl enable mediamap
sudo systemctl start mediamap
sudo systemctl restart nginx

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at: http://$(curl -s ifconfig.me)"
echo "📊 Check status with: sudo systemctl status mediamap" 