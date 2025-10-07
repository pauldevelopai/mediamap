#!/usr/bin/env python3
"""
Script to trigger South African doctor scraping for HealthPIN
"""
import sys
import os
import requests
import json

# Add the backend path
sys.path.append('/opt/mediamap/backend')

def trigger_doctor_scraping():
    """Trigger doctor scraping via direct method call"""
    try:
        # Set up Flask app context
        from backend.app import app
        
        with app.app_context():
            from agents.healthpin_agent import HealthPINAgent
            from agents.base_agent import AgentConfig
            from backend.healthpin.models import Doctor
            from backend.models import db
            
            print("🔍 Setting up HealthPIN agent...")
            
            # Create config
            config = AgentConfig(
                name='healthpin',
                section='healthpin', 
                data_sources=[
                    'https://www.who.int/rss-feeds/news-english.xml',
                    'https://www.health.harvard.edu/rss'
                ],
                learning_interval=3600,
                max_data_points=1000,
                api_keys={},
                storage_path='/opt/mediamap/backend/agents/storage/healthpin'
            )
            
            # Initialize agent without creating files
            print("🤖 Initializing agent...")
            
            # Create a minimal agent instance
            class MinimalHealthPINAgent:
                def __init__(self):
                    pass
                    
                def scrape_doctors_south_africa(self, limit=20):
                    """Scrape South African doctors from OpenStreetMap"""
                    import requests
                    import time
                    from datetime import datetime
                    
                    print(f"🌍 Scraping South African doctors (limit: {limit})...")
                    
                    # Overpass API query for healthcare facilities in South Africa
                    overpass_url = "http://overpass-api.de/api/interpreter"
                    query = """
                    [out:json][timeout:25];
                    (
                      node["amenity"="doctors"]["addr:country"="ZA"];
                      node["amenity"="clinic"]["addr:country"="ZA"];
                      node["healthcare"="doctor"]["addr:country"="ZA"];
                    );
                    out body;
                    """
                    
                    try:
                        response = requests.post(overpass_url, data=query, timeout=30)
                        response.raise_for_status()
                        data = response.json()
                        
                        doctors_added = 0
                        elements = data.get('elements', [])[:limit]
                        
                        print(f"📊 Found {len(elements)} healthcare facilities")
                        
                        for element in elements:
                            tags = element.get('tags', {})
                            
                            # Extract doctor information
                            name = (tags.get('name') or 
                                   tags.get('healthcare:speciality') or 
                                   tags.get('amenity', 'Healthcare Provider'))
                            
                            # Get location info
                            city = (tags.get('addr:city') or 
                                   tags.get('addr:suburb') or 
                                   'Unknown City')
                            
                            province = (tags.get('addr:state') or 
                                       tags.get('addr:province') or 
                                       'Unknown Province')
                            
                            # Get coordinates
                            lat = element.get('lat', 0)
                            lon = element.get('lon', 0)
                            
                            # Create doctor entry
                            try:
                                # Check if doctor already exists
                                existing = Doctor.query.filter_by(
                                    name=name, 
                                    city=city, 
                                    province=province
                                ).first()
                                
                                if not existing:
                                    doctor = Doctor(
                                        name=name,
                                        specialties=['General Practice'],
                                        city=city,
                                        province=province,
                                        latitude=lat,
                                        longitude=lon,
                                        practice_name=tags.get('operator', 'Private Practice'),
                                        phone=tags.get('phone', ''),
                                        website=tags.get('website', ''),
                                        is_verified=True,
                                        created_at=datetime.utcnow()
                                    )
                                    
                                    db.session.add(doctor)
                                    doctors_added += 1
                                    
                                    if doctors_added % 5 == 0:
                                        print(f"✅ Added {doctors_added} doctors so far...")
                                        
                            except Exception as e:
                                print(f"⚠️  Error adding doctor {name}: {e}")
                                continue
                        
                        # Commit all changes
                        db.session.commit()
                        
                        result = {
                            'success': True,
                            'doctors_found': len(elements),
                            'doctors_added': doctors_added,
                            'message': f'Successfully scraped {doctors_added} South African doctors'
                        }
                        
                        print(f"🎉 {result['message']}")
                        return result
                        
                    except requests.RequestException as e:
                        error_msg = f"Failed to fetch data from Overpass API: {e}"
                        print(f"❌ {error_msg}")
                        return {'success': False, 'error': error_msg}
                    
                    except Exception as e:
                        error_msg = f"Error during doctor scraping: {e}"
                        print(f"❌ {error_msg}")
                        return {'success': False, 'error': error_msg}
            
            # Create and run the agent
            agent = MinimalHealthPINAgent()
            result = agent.scrape_doctors_south_africa(limit=50)
            
            # Check final count
            final_count = Doctor.query.count()
            print(f"📊 Total doctors now in database: {final_count}")
            
            if final_count > 0:
                print("👨‍⚕️ Sample doctors:")
                sample_doctors = Doctor.query.limit(3).all()
                for doc in sample_doctors:
                    print(f"  - {doc.name} in {doc.city}, {doc.province}")
            
            return result
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    print("🏥 SOUTH AFRICAN DOCTOR SCRAPING")
    print("=" * 40)
    result = trigger_doctor_scraping()
    print("\n" + "=" * 40)
    print(f"📋 Final result: {result}")
