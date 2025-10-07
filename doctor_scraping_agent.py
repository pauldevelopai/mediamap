#!/usr/bin/env python3
"""
HealthPIN AI Agent - South African Doctor Scraping
Direct execution script for Lightsail instance
"""
import sys
import os
import json
import requests
from datetime import datetime

# Add backend path
sys.path.append('/opt/mediamap/backend')

def run_ai_doctor_scraping():
    """Run the AI agent to scrape South African doctors"""
    print("🤖 HEALTHPIN AI AGENT - DOCTOR SCRAPING")
    print("=" * 50)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import Flask app and set up context
        from backend.app import app
        
        with app.app_context():
            print("✅ Flask app context established")
            
            # Import required modules
            from backend.agents.agent_manager import agent_manager
            from backend.healthpin.models import Doctor
            from backend.models import db
            
            print("✅ Modules imported successfully")
            
            # Check initial doctor count
            initial_count = Doctor.query.count()
            print(f"📊 Initial doctors in database: {initial_count}")
            print()
            
            # Get the HealthPIN agent
            if 'healthpin' not in agent_manager.agents:
                print("❌ HealthPIN agent not found in agent manager")
                return False
            
            agent = agent_manager.agents['healthpin']
            print("✅ HealthPIN AI agent retrieved")
            
            # Check if scraping method exists
            if not hasattr(agent, 'scrape_doctors_south_africa'):
                print("❌ Doctor scraping method not available")
                return False
            
            print("✅ Doctor scraping method confirmed")
            print()
            
            # Define progress callback
            def progress_callback(percent, message):
                print(f"🔄 Progress: {percent}% - {message}")
            
            # Trigger the AI agent scraping
            print("🚀 Triggering AI agent doctor scraping...")
            print("🌍 Searching OpenStreetMap for South African healthcare facilities...")
            print()
            
            try:
                result = agent.scrape_doctors_south_africa(
                    limit=100, 
                    progress_cb=progress_callback
                )
                
                print()
                print("📋 SCRAPING RESULTS:")
                print("-" * 30)
                
                if result and result.get('success'):
                    print("✅ AI Agent scraping completed successfully!")
                    
                    if 'doctors_found' in result:
                        print(f"🔍 Healthcare facilities found: {result['doctors_found']}")
                    
                    if 'doctors_added' in result:
                        print(f"👨‍⚕️ New doctors added to database: {result['doctors_added']}")
                    
                    if 'message' in result:
                        print(f"📝 Agent message: {result['message']}")
                        
                else:
                    print("⚠️  Scraping completed with issues")
                    if result and 'error' in result:
                        print(f"🚨 Error: {result['error']}")
                
            except Exception as scrape_error:
                print(f"❌ Error during scraping: {scrape_error}")
                return False
            
            print()
            
            # Check final count
            final_count = Doctor.query.count()
            print(f"📊 Final doctors in database: {final_count}")
            
            if final_count > initial_count:
                added = final_count - initial_count
                print(f"🎉 SUCCESS! Added {added} new South African doctors!")
                
                # Show sample doctors
                print()
                print("👨‍⚕️ Sample of scraped doctors:")
                sample_doctors = Doctor.query.limit(5).all()
                for i, doc in enumerate(sample_doctors, 1):
                    specialties = ', '.join(doc.specialties) if doc.specialties else 'General Practice'
                    print(f"  {i}. {doc.name}")
                    print(f"     📍 {doc.city}, {doc.province}")
                    print(f"     🏥 {specialties}")
                    if doc.phone:
                        print(f"     📞 {doc.phone}")
                    print()
                
            elif final_count == initial_count:
                print("ℹ️  No new doctors added (may already exist in database)")
            
            print()
            print("🌐 View results at: http://35.177.61.112/healthpin/doctors")
            print()
            
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running this from the correct directory with venv activated")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check if we're in the right directory
    if not os.path.exists('/opt/mediamap/backend'):
        print("❌ Not running from correct directory")
        return False
    
    # Check if virtual environment is activated
    if 'venv' not in sys.executable:
        print("⚠️  Virtual environment may not be activated")
        print("💡 Run: source venv/bin/activate")
    
    print("✅ Prerequisites check passed")
    return True

def main():
    """Main execution function"""
    if not check_prerequisites():
        return
    
    print()
    success = run_ai_doctor_scraping()
    
    print()
    print("=" * 50)
    print(f"🕒 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("🎉 AI Agent doctor scraping completed successfully!")
        print()
        print("Next steps:")
        print("1. Visit http://35.177.61.112/healthpin/doctors")
        print("2. You should now see real South African doctors")
        print("3. Click on individual doctors for more details")
    else:
        print("❌ AI Agent scraping failed")
        print()
        print("Troubleshooting:")
        print("1. Check that the HealthPIN agent is running")
        print("2. Verify database permissions")
        print("3. Check network connectivity to OpenStreetMap API")

if __name__ == "__main__":
    main()
