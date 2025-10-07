#!/usr/bin/env python3
"""
Automated AI Agent Doctor Scraping for HealthPIN
Triggers the HealthPIN AI agent to scrape South African doctors
"""
import requests
import json
import time
from datetime import datetime

def get_admin_session():
    """Get authenticated session for admin user"""
    session = requests.Session()
    
    # Login as admin
    login_url = "http://35.177.61.112/login"
    
    # Get login page to get any CSRF tokens
    login_page = session.get(login_url)
    
    # Login with admin credentials
    login_data = {
        'username': 'admin',
        'password': 'admin123',  # Default admin password
        'submit': 'Sign In'
    }
    
    response = session.post(login_url, data=login_data, allow_redirects=True)
    
    if response.status_code == 200 and 'dashboard' in response.url.lower():
        print("✅ Successfully logged in as admin")
        return session
    else:
        print(f"❌ Login failed. Status: {response.status_code}")
        print(f"Response URL: {response.url}")
        return None

def trigger_ai_doctor_scraping(session, limit=50):
    """Trigger the AI agent to scrape doctors"""
    scrape_url = "http://35.177.61.112/healthpin/scrape-doctors"
    
    print(f"🤖 Triggering AI agent to scrape {limit} South African doctors...")
    
    payload = {"limit": limit}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = session.post(scrape_url, json=payload, headers=headers, timeout=120)
        
        if response.status_code == 200:
            try:
                result = response.json()
                return result
            except json.JSONDecodeError:
                print("⚠️  Response received but not JSON format")
                return {"success": False, "error": "Invalid JSON response"}
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except requests.Timeout:
        print("⏱️  Request timed out - scraping may still be running in background")
        return {"success": "timeout", "message": "Scraping started but timed out"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

def check_doctor_count():
    """Check how many doctors are now in the database"""
    try:
        # Simple API call to check doctor count
        response = requests.get("http://35.177.61.112/healthpin/doctors", timeout=10)
        
        if "0 verified healthcare professionals" in response.text:
            return 0
        elif "verified healthcare professionals" in response.text:
            # Try to extract the number
            import re
            match = re.search(r'(\d+)\s+verified healthcare professionals', response.text)
            if match:
                return int(match.group(1))
        
        return "unknown"
    except Exception as e:
        print(f"Error checking doctor count: {e}")
        return "error"

def main():
    print("🏥 AI AGENT DOCTOR SCRAPING")
    print("=" * 50)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check initial doctor count
    initial_count = check_doctor_count()
    print(f"📊 Initial doctor count: {initial_count}")
    print()
    
    # Get authenticated session
    print("🔐 Authenticating as admin...")
    session = get_admin_session()
    
    if not session:
        print("❌ Failed to authenticate. Cannot proceed.")
        return
    
    print()
    
    # Trigger AI agent scraping
    print("🤖 Triggering HealthPIN AI Agent...")
    result = trigger_ai_doctor_scraping(session, limit=100)
    
    print()
    print("📋 SCRAPING RESULT:")
    print("-" * 30)
    
    if result.get("success") == True:
        print("✅ AI Agent scraping completed successfully!")
        if "doctors_added" in result:
            print(f"👨‍⚕️ Doctors added: {result['doctors_added']}")
        if "doctors_found" in result:
            print(f"🔍 Doctors found: {result['doctors_found']}")
        if "message" in result:
            print(f"📝 Message: {result['message']}")
            
    elif result.get("success") == "timeout":
        print("⏱️  Scraping started but response timed out")
        print("🔄 The AI agent may still be working in the background")
        
    else:
        print("❌ Scraping failed or encountered issues")
        if "error" in result:
            print(f"🚨 Error: {result['error']}")
    
    print()
    
    # Wait a moment and check final count
    print("⏳ Waiting 10 seconds then checking results...")
    time.sleep(10)
    
    final_count = check_doctor_count()
    print(f"📊 Final doctor count: {final_count}")
    
    if isinstance(initial_count, int) and isinstance(final_count, int):
        added = final_count - initial_count
        if added > 0:
            print(f"🎉 SUCCESS! Added {added} new doctors!")
        elif added == 0:
            print("ℹ️  No new doctors added (may already exist or scraping in progress)")
        else:
            print("⚠️  Unexpected count change")
    
    print()
    print("🌐 Visit the doctors page to see results:")
    print("   http://35.177.61.112/healthpin/doctors")
    print()
    print("=" * 50)
    print(f"🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
