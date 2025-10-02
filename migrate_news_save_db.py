#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.app import app, db
from backend.models import SavedNews

def migrate_news_save_database():
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database migration completed successfully!")
            print("✅ SavedNews table created")
            
            # Check if SavedNews table exists
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            if 'saved_news' in inspector.get_table_names():
                print("✅ SavedNews table verified in database")
            else:
                print("❌ SavedNews table not found in database")
                return False
                
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    return True

if __name__ == "__main__":
    migrate_news_save_database() 