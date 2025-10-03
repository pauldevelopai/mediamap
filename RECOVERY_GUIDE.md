# Lightsail Recovery & Fix Deployment Guide

## 🚨 Current Situation
- Lightsail instance crashed (UPSTREAM_ERROR 515)
- Instance is being force-stopped (can take 5-10 minutes)
- Fixes are ready and committed to GitHub

## 🔄 Recovery Steps

### Step 1: Wait for Instance to Stop
- **Current status**: Stopping (started at 09:10 CEST)
- **Expected time**: 5-10 minutes for force stop
- **Don't interrupt**: Let it complete the stop process

### Step 2: Start Instance
1. Once stopped, click **"Start"** in Lightsail Console
2. Wait 3-5 minutes for full boot
3. **Check if IP changed** - note new IP if different

### Step 3: Test Connection
```bash
ssh -i LightsailDefaultKey-eu-west-2.pem ubuntu@35.177.61.112 "echo 'Connected!'"
```

### Step 4: Deploy Fixes (Choose One Method)

#### Method A: Quick Git Pull (Recommended)
```bash
ssh -i LightsailDefaultKey-eu-west-2.pem ubuntu@35.177.61.112 "
cd /opt/mediamap && 
git pull origin main && 
python fix_all_database_issues.py && 
python enable_healthpin_primary.py
"
```

#### Method B: Automated Script
```bash
./deploy_fixes.sh
```

## 🎯 What Gets Fixed

### 1. Organisations Table Error
- **Problem**: `no such table: organisations`
- **Fix**: Creates all missing SQLAlchemy tables
- **Result**: Organisation search works without errors

### 2. HealthPIN Doctor Scraping
- **Problem**: Doctor scraping agent not configured as primary
- **Fix**: Enables OpenStreetMap doctor scraping for South Africa
- **Result**: HealthPIN becomes primary medical data agent

## 🔍 Verification Commands

After deployment, verify fixes:

```bash
# Test organisations table
sqlite3 instance/aimap.db "SELECT COUNT(*) FROM organisations;"

# Test HealthPIN agent
curl -X POST http://localhost:3000/agents/healthpin/scrape/doctors \
  -H "Content-Type: application/json" \
  -d '{"limit": 5}'
```

## 🚨 If Recovery Fails

### Option 1: Create New Instance from Snapshot
1. Create snapshot of current instance (preserves data)
2. Create new instance from snapshot
3. Deploy fixes to new instance

### Option 2: Manual Database Recreation
1. Backup existing data: `cp instance/aimap.db instance/aimap.db.backup`
2. Run database recreation scripts
3. Restore data if needed

## 📞 Next Steps After Recovery
1. ✅ Verify organisation search works
2. ✅ Test HealthPIN doctor scraping
3. ✅ Confirm all services running
4. 📝 Document any additional issues found

---
**Created**: October 3, 2025 09:10 CEST  
**Status**: Instance stopping, fixes ready for deployment
