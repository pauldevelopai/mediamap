#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app import app, db
from backend.models import NotionIntegration

def migrate_database():
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            print("✅ Database migration completed successfully!")
            print("✅ NotionIntegration table created")
            
            # Check if tables exist
            tables = db.engine.table_names()
            print(f"📊 Current tables: {tables}")
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    
    return True

if __name__ == "__main__":
    migrate_database() 