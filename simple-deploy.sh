#!/bin/bash

# Simple deployment script for DataSafe
echo "🚀 Starting simple DataSafe deployment..."

# Update system
sudo apt update

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv nginx

# Create virtual environment
cd /opt/datasafe
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/datasafe.service > /dev/null <<EOF
[Unit]
Description=DataSafe Flask Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/datasafe
Environment=PATH=/opt/datasafe/venv/bin
ExecStart=/opt/datasafe/venv/bin/gunicorn --bind 0.0.0.0:8000 backend.app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Configure nginx
sudo tee /etc/nginx/sites-available/datasafe > /dev/null <<EOF
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
sudo ln -sf /etc/nginx/sites-available/datasafe /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Start services
sudo systemctl daemon-reload
sudo systemctl enable datasafe
sudo systemctl start datasafe
sudo systemctl restart nginx

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at: http://$(curl -s ifconfig.me)"
echo "📊 Check status with: sudo systemctl status datasafe" 