# 🔧 Lightsail Troubleshooting Guide

## 🚨 Current Issue: SSH Connection Timeout

**Problem:** SSH connection to `35.176.169.218` is timing out
**Status:** Instance shows as "Running" in AWS Console
**Region:** eu-west-2a (London)

## 🔍 Diagnostic Steps

### 1. Network Connectivity Test
```bash
# Test basic connectivity
ping -c 3 35.176.169.218

# Test SSH port specifically
nc -zv 35.176.169.218 22
```

### 2. SSH Key Verification
```bash
# Check key file exists and permissions
ls -la LightsailDefaultKey-eu-west-2.pem

# Verify key format
head -1 LightsailDefaultKey-eu-west-2.pem
# Should show: -----BEGIN OPENSSH PRIVATE KEY----- or -----BEGIN RSA PRIVATE KEY-----
```

### 3. AWS Console Actions
Since SSH is not working, use the AWS Console to:

1. **Check Instance Status**
   - Go to Lightsail Console
   - Verify instance is "Running"
   - Check CPU/Memory usage

2. **Check Networking**
   - Go to Networking tab
   - Verify SSH (port 22) is open in firewall
   - Check if any custom firewall rules exist

3. **Check Instance Logs**
   - Use "Connect using browser" option
   - Check system logs for errors

## 🛠️ Solutions to Try

### Option 1: Restart Instance
1. Go to AWS Lightsail Console
2. Select your instance
3. Click "Stop" → Wait → Click "Start"
4. Wait 2-3 minutes for full startup
5. Try SSH again

### Option 2: Check Firewall Rules
1. In Lightsail Console → Networking
2. Ensure SSH (port 22) is open
3. Add rule if missing:
   - Protocol: TCP
   - Port: 22
   - Source: Anywhere (0.0.0.0/0)

### Option 3: Use Browser-Based SSH
1. In Lightsail Console
2. Click "Connect using browser"
3. This bypasses network issues
4. Once connected, check SSH service:
   ```bash
   sudo systemctl status ssh
   sudo systemctl restart ssh
   ```

### Option 4: Create New Instance
If the current instance is corrupted:
1. Create snapshot of current instance
2. Create new instance from snapshot
3. Update IP address in scripts

## 🔄 Alternative Access Methods

### Browser SSH (Recommended)
```bash
# Use AWS Console → Connect using browser
# Then run these commands:
cd /opt/mediamap
sudo systemctl status mediamap
sudo journalctl -u mediamap -f
```

### Application Status Check
Even if SSH doesn't work, the application might still be running:
```bash
# Test application directly
curl -I http://35.176.169.218:8000
curl -I http://35.176.169.218:3000
```

## 📋 Current Application Status

Based on our previous work:
- ✅ **Database tables created** and working
- ✅ **Admin user exists** (admin/admin123)
- ✅ **Application was running** on port 3000
- ⚠️ **SSH access blocked** (network issue)

## 🚀 Next Steps

### Immediate Actions:
1. **Use AWS Console** to access instance via browser
2. **Check application status** using browser SSH
3. **Verify firewall rules** in Lightsail Console
4. **Test application URLs** directly

### If Application is Running:
- Test: http://35.176.169.218:8000
- Test: http://35.176.169.218:3000
- Login: admin / admin123

### If SSH Still Doesn't Work:
1. **Restart instance** via AWS Console
2. **Check firewall rules** 
3. **Create new instance** if needed
4. **Update scripts** with new IP address

## 🔧 Script Updates Needed

If you get a new IP address, update these files:
- `connect-lightsail.sh` (line 10: LIGHTSAIL_IP)
- `update-lightsail.sh` (line 10: LIGHTSAIL_IP)
- `push-to-github.sh` (line 10: LIGHTSAIL_IP)
- `lightsail-manager.sh` (line 35: LIGHTSAIL_IP)

## 💡 Pro Tips

1. **Always use browser SSH** when direct SSH fails
2. **Check application URLs** even if SSH doesn't work
3. **Restart instance** is often the quickest fix
4. **Firewall rules** are the most common cause
5. **Keep backups** before making changes

---

**Current Priority:** Use AWS Console browser SSH to check application status and fix SSH access! 🚀
