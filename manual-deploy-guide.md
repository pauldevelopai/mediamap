# Manual Lightsail Deployment Guide

## 🎯 Your Instance is Running!
**IP Address:** 35.176.169.218:8000

## 📋 Manual Deployment Steps

### 1. Access Lightsail Console
- Go to: https://lightsail.aws.amazon.com/ls/webapp/eu-west-2/instances
- Find your instance
- Click on it

### 2. Use Browser-Based SSH
- Click "Connect using SSH" or "Terminal" button
- This opens a browser-based SSH session

### 3. Upload and Deploy Files
Run these commands in the browser SSH:

```bash
# Navigate to the app directory
cd /opt/mediamap

# Create backup
sudo cp -r backend/templates backend/templates.backup

# Download the updated templates
sudo wget -O backend/templates/user_dashboard.html "https://raw.githubusercontent.com/your-repo/main/backend/templates/user_dashboard.html"
sudo wget -O backend/templates/user_chats.html "https://raw.githubusercontent.com/your-repo/main/backend/templates/user_chats.html"  
sudo wget -O backend/templates/login.html "https://raw.githubusercontent.com/your-repo/main/backend/templates/login.html"

# Set permissions
sudo chown ubuntu:ubuntu backend/templates/*.html
sudo chmod 644 backend/templates/*.html

# Restart the application
docker-compose restart

# Wait for restart
sleep 15

# Check status
docker-compose ps
```

### 4. Alternative: Manual File Upload
If the wget method doesn't work:

1. **Download the files locally:**
   - `backend/templates/user_dashboard.html`
   - `backend/templates/user_chats.html` 
   - `backend/templates/login.html`

2. **Upload via Lightsail console:**
   - Use the file upload feature in the browser SSH
   - Or copy/paste the content manually

3. **Restart the app:**
   ```bash
   cd /opt/mediamap
   docker-compose restart
   ```

## ✅ What Will Be Updated

- **Username display** in top bar
- **Working chat functionality** (fixed 500 error)
- **Conversation saving indicator**
- **Feedback system** on my-chats page
- **Modern login page**
- **All design improvements**

## 🌐 Test Your Updates

After deployment, visit:
- **Main app:** http://35.176.169.218:8000
- **User dashboard:** http://35.176.169.218:8000/user-dashboard
- **My chats:** http://35.176.169.218:8000/my-chats

## 🔧 If You Need Help

The local version is working perfectly at http://localhost:8000 - you can test all features there while we get the Lightsail version updated! 