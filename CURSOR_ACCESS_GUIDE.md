# 🎯 Cursor Access Guide for Lightsail

## 🚀 Quick Setup

### 1. **One-Time Clean Deploy**
```bash
./clean-deploy.sh
```
This creates a perfect mirror of your local machine on Lightsail.

### 2. **Easy SSH Access from Cursor**

#### Option A: Using SSH Config (Recommended)
```bash
# From Cursor terminal:
ssh lightsail
# or
ssh mediamap
```

#### Option B: Direct SSH
```bash
ssh -i "LightsailDefaultKey-eu-west-2.pem" ubuntu@35.177.61.112
```

### 3. **Quick Sync for Development**
```bash
# Sync changes only
./sync-to-lightsail.sh

# Sync and restart application
./sync-to-lightsail.sh --restart
```

## 🔧 Development Workflow

### **Daily Development:**
1. **Make changes locally** in Cursor
2. **Sync to Lightsail:** `./sync-to-lightsail.sh`
3. **Test on server:** http://35.177.61.112:8000
4. **SSH for debugging:** `ssh lightsail`

### **When You Need to Restart:**
```bash
./sync-to-lightsail.sh --restart
```

### **When You Need Full Deploy:**
```bash
./clean-deploy.sh
```

## 🌐 Application Access

- **Main App:** http://35.177.61.112:8000
- **Alternative:** http://35.177.61.112:3000
- **Admin Login:** admin / admin123

## 📋 Useful Commands

### **From Local Machine:**
```bash
# Connect to Lightsail
ssh lightsail

# Quick sync
./sync-to-lightsail.sh

# Full deployment
./clean-deploy.sh

# Check status
./lightsail-manager.sh
```

### **From Lightsail (via SSH):**
```bash
# Navigate to app
cd /opt/mediamap

# Check application status
sudo systemctl status mediamap

# View logs
sudo journalctl -u mediamap -f

# Restart application
sudo systemctl restart mediamap

# Check database
sudo -u www-data python3 -c "
import sqlite3
conn = sqlite3.connect('/opt/mediamap/instance/media_analysis.db')
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\"')
tables = cursor.fetchall()
for table in tables:
    print(table[0])
conn.close()
"
```

## 🔄 Port Forwarding

The SSH config includes port forwarding:
- **Local 8000** → **Lightsail 8000**
- **Local 3000** → **Lightsail 3000**

This means you can access the app at:
- http://localhost:8000 (when SSH is connected)
- http://localhost:3000 (when SSH is connected)

## 🛠️ Troubleshooting

### **Connection Issues:**
```bash
# Test connection
ssh -i "LightsailDefaultKey-eu-west-2.pem" -o ConnectTimeout=10 ubuntu@35.177.61.112 "echo 'test'"

# Check instance status
curl -I http://35.177.61.112:8000
```

### **Application Issues:**
```bash
# SSH to server
ssh lightsail

# Check logs
sudo journalctl -u mediamap -n 20

# Restart service
sudo systemctl restart mediamap
```

### **Database Issues:**
```bash
# SSH to server
ssh lightsail

# Check database
cd /opt/mediamap
sudo -u www-data python3 -c "
import sqlite3
conn = sqlite3.connect('instance/media_analysis.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM users')
print('Users:', cursor.fetchone()[0])
conn.close()
"
```

## 📁 File Structure

```
/opt/mediamap/          # Application root
├── backend/            # Backend code
├── venv/              # Python virtual environment
├── instance/          # Database files
│   └── media_analysis.db
├── requirements.txt   # Python dependencies
└── gunicorn.conf.py   # Gunicorn configuration
```

## 🎯 Pro Tips

1. **Use SSH config** for easy access: `ssh lightsail`
2. **Quick sync** for development: `./sync-to-lightsail.sh`
3. **Port forwarding** lets you use localhost URLs
4. **Check logs** when debugging: `sudo journalctl -u mediamap -f`
5. **Database access** via Python scripts on server

## 🚨 Emergency Commands

```bash
# Stop everything
ssh lightsail "sudo systemctl stop mediamap && sudo pkill -f python"

# Clean restart
./clean-deploy.sh

# Check what's running
ssh lightsail "ps aux | grep -E '(python|gunicorn)' | grep -v grep"
```

---

**Ready to develop? Run `./clean-deploy.sh` and then `ssh lightsail`! 🚀**






