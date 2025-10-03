#!/usr/bin/env python3
"""Fix the organisations table issue on Lightsail"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from flask import Flask
from backend.models import db

# Create minimal Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/aimap.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def fix_database():
    """Create all database tables"""
    with app.app_context():
        try:
            # Import all models to ensure they're registered
            from backend.models import User, MediaAnalysis, Chat
            from backend.aimap.models import (
                Organisation, Lead, LeadActivity, Interaction,
                ResearchReport, CustomData, ConsultingProject, ProjectMilestone
            )
            
            print("Creating database tables...")
            db.create_all()
            
            print("✅ Database tables created successfully!")
            
            # Verify organisations table exists
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            print(f"\nAvailable tables: {', '.join(tables)}")
            
            if 'organisations' in tables:
                print("✅ organisations table verified")
            else:
                print("❌ organisations table NOT found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    fix_database()



