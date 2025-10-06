#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from flask import Flask
from backend.models import db

app = Flask(__name__)
db_path = "/opt/mediamap/instance/aimap.db"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    try:
        # Create HealthPIN doctor tables
        from backend.healthpin.models import Doctor, Patient
        
        print("Creating HealthPIN doctor tables...")
        Doctor.__table__.create(db.engine, checkfirst=True)
        Patient.__table__.create(db.engine, checkfirst=True)
        print("HealthPIN tables created successfully!")
        
        # Test the HealthPIN agent
        from backend.agents.agent_manager import AgentManager
        
        print("Initializing HealthPIN agent...")
        agent_manager = AgentManager()
        
        if 'healthpin' in agent_manager.agents:
            agent = agent_manager.agents['healthpin']
            print("SUCCESS: HealthPIN agent found!")
            
            if hasattr(agent, 'scrape_doctors_south_africa'):
                print("SUCCESS: Doctor scraping method available!")
                
                # Test with small limit
                print("Testing doctor scraping (limit=2)...")
                result = agent.scrape_doctors_south_africa(limit=2)
                
                if result.get('success'):
                    print(f"SUCCESS: Doctor scraping test passed!")
                    print(f"  Created: {result.get('created', 0)}")
                    print(f"  Updated: {result.get('updated', 0)}")
                else:
                    print(f"WARNING: {result.get('error', 'Unknown error')}")
            else:
                print("ERROR: Doctor scraping method not found")
        else:
            print("ERROR: HealthPIN agent not found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
