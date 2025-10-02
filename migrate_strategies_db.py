#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.app import app, db
from backend.models import SavedStrategy

def migrate_strategies_database():
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database migration completed successfully!")
            print("✅ SavedStrategy table created")
            
            # Check if SavedStrategy table exists
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            if 'saved_strategies' in inspector.get_table_names():
                print("✅ SavedStrategy table verified in database")
            else:
                print("❌ SavedStrategy table not found in database")
                return False
                
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    return True

if __name__ == "__main__":
    migrate_strategies_database() 