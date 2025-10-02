#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.app import app, db
from backend.models import News

def migrate_news_database():
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database migration completed successfully!")
            print("✅ News table created")
            
            # Check if News table exists
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            if 'news' in inspector.get_table_names():
                print("✅ News table verified in database")
            else:
                print("❌ News table not found in database")
                return False
                
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    return True

if __name__ == "__main__":
    migrate_news_database() 