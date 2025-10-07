#!/usr/bin/env python3
"""
Database migration script to add the strategies table
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys
import os
sys.path.append('/opt/datasafe')

from backend.app import app, db
from backend.strategies_crawler import StrategyEntry

def migrate_strategies_table():
    """Create the strategies table if it doesn't exist"""
    with app.app_context():
        try:
            # Create the strategies table
            db.create_all()
            print("✅ Strategies table created successfully")
            
            # Check if table exists
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            if 'strategies' in inspector.get_table_names():
                print("✅ Strategies table verified in database")
            else:
                print("❌ Strategies table not found in database")
                
        except Exception as e:
            print(f"❌ Error creating strategies table: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("Starting strategies table migration...")
    success = migrate_strategies_table()
    if success:
        print("🎉 Migration completed successfully!")
    else:
        print("💥 Migration failed!")
        sys.exit(1) 