#!/usr/bin/env python3
"""Create database tables for session management"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import UserSession, Memory

def create_tables():
    """Create the new database tables"""
    with app.app_context():
        try:
            # Create the new tables
            db.create_all()
            print("✅ Database tables created successfully!")
            
            # Check if tables exist
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'user_session' in tables:
                print("✅ user_session table created")
            if 'memory' in tables:
                print("✅ memory table created")
                
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            return False
    
    return True

if __name__ == "__main__":
    create_tables()
