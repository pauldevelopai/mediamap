#!/usr/bin/env python3
"""
MediaMap Health Check Script
===========================

Quick health check after instance recovery to verify all fixes are working.
"""

import os
import sys
import sqlite3
import requests
import json
from pathlib import Path

def check_database():
    """Check if database tables exist and are accessible"""
    print("🔍 Checking database health...")
    
    db_path = Path(__file__).parent / 'instance' / 'aimap.db'
    
    if not db_path.exists():
        print("❌ Database file not found")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check organisations table
        cursor.execute("SELECT COUNT(*) FROM organisations")
        org_count = cursor.fetchone()[0]
        print(f"✅ Organisations table: {org_count} records")
        
        # Check HealthPIN tables
        healthpin_tables = [
            'healthpin_patients',
            'healthpin_doctors', 
            'healthpin_health_records',
            'healthpin_doctor_matches'
        ]
        
        for table in healthpin_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"✅ {table}: {count} records")
            except sqlite3.OperationalError:
                print(f"⚠️ {table}: Table not found (will be created on first use)")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_web_service():
    """Check if the web service is running"""
    print("\n🌐 Checking web service...")
    
    try:
        response = requests.get('http://localhost:3000', timeout=10)
        if response.status_code == 200:
            print("✅ Web service is running")
            return True
        else:
            print(f"⚠️ Web service returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Web service not responding (may still be starting)")
        return False
    except Exception as e:
        print(f"❌ Web service error: {e}")
        return False

def check_healthpin_agent():
    """Check if HealthPIN agent is available"""
    print("\n🏥 Checking HealthPIN agent...")
    
    try:
        # Check agent status endpoint
        response = requests.get('http://localhost:3000/agents/healthpin/doctor-directory/status', timeout=10)
        if response.status_code == 200:
            print("✅ HealthPIN agent is available")
            return True
        else:
            print(f"⚠️ HealthPIN agent returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ HealthPIN agent not responding")
        return False
    except Exception as e:
        print(f"❌ HealthPIN agent error: {e}")
        return False

def check_system_resources():
    """Check system resources"""
    print("\n💻 Checking system resources...")
    
    try:
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage('/')
        free_gb = free // (1024**3)
        print(f"✅ Disk space: {free_gb}GB free")
        
        # Check if processes are running
        import subprocess
        result = subprocess.run(['pgrep', '-f', 'python.*app.py'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Python app process running")
        else:
            print("⚠️ Python app process not found")
        
        return True
        
    except Exception as e:
        print(f"❌ System check error: {e}")
        return False

def main():
    """Run all health checks"""
    print("🚀 MediaMap Health Check")
    print("=" * 40)
    
    checks = [
        ("Database", check_database),
        ("Web Service", check_web_service), 
        ("HealthPIN Agent", check_healthpin_agent),
        ("System Resources", check_system_resources)
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 HEALTH CHECK SUMMARY")
    print("=" * 40)
    
    all_good = True
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name}: {'PASS' if status else 'FAIL'}")
        if not status:
            all_good = False
    
    if all_good:
        print("\n🎉 All systems healthy!")
        print("💡 MediaMap is ready to use")
    else:
        print("\n⚠️ Some issues detected")
        print("💡 Check the details above and run fixes if needed")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
