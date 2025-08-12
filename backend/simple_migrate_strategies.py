#!/usr/bin/env python3
"""
Simple database migration script to add the strategies table
"""

import os
import sys
sys.path.insert(0, '/opt/datasafe')

# Set environment variables
os.environ['FLASK_APP'] = 'backend.app'
os.environ['FLASK_ENV'] = 'production'
os.environ['PYTHONPATH'] = '/opt/datasafe'

from backend.app import app, db
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class StrategyEntry(Base):
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    source = Column(String(200), nullable=False)
    url = Column(String(500), nullable=False)
    use_cases = Column(Text)  # JSON string
    code_examples = Column(Text)  # JSON string
    implementation_steps = Column(Text)  # JSON string
    ai_insights = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

def migrate_strategies_table():
    """Create the strategies table if it doesn't exist"""
    with app.app_context():
        try:
            # Create the strategies table
            StrategyEntry.__table__.create(db.engine, checkfirst=True)
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