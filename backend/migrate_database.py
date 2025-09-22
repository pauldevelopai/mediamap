#!/usr/bin/env python3
"""
Database Migration Script
Handles database schema changes and migrations properly
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError

# Add the backend directory to Python path
basedir = os.path.abspath(os.path.dirname(__file__))
if basedir not in sys.path:
    sys.path.insert(0, basedir)

from models import db, User
from flask import Flask

def create_app():
    """Create Flask app for migrations"""
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "instance", "media_analysis.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    return app

def check_column_exists(engine, table_name, column_name):
    """Check if a column exists in a table"""
    try:
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception:
        return False

def add_column_if_not_exists(engine, table_name, column_name, column_type):
    """Add a column to a table if it doesn't exist"""
    if not check_column_exists(engine, table_name, column_name):
        try:
            with engine.connect() as conn:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))
                conn.commit()
                print(f"✅ Added column {column_name} to {table_name}")
                return True
        except Exception as e:
            print(f"❌ Error adding column {column_name} to {table_name}: {e}")
            return False
    else:
        print(f"ℹ️  Column {column_name} already exists in {table_name}")
        return True

def run_migrations():
    """Run database migrations"""
    app = create_app()
    
    with app.app_context():
        engine = db.engine
        
        print("🔄 Running database migrations...")
        
        # Migration 1: Add missing columns to users table
        print("\n📋 Migration 1: User table columns")
        user_migrations = [
            ('is_admin', 'BOOLEAN DEFAULT 0'),
            ('last_login', 'DATETIME'),
            ('latitude', 'FLOAT'),
            ('longitude', 'FLOAT'),
            ('location_name', 'VARCHAR(200)')
        ]
        
        for column_name, column_type in user_migrations:
            add_column_if_not_exists(engine, 'users', column_name, column_type)
        
        # Migration 2: Create tables if they don't exist
        print("\n📋 Migration 2: Create missing tables")
        try:
            db.create_all()
            print("✅ All tables created/verified")
        except Exception as e:
            print(f"⚠️  Warning creating tables: {e}")
        
        print("\n✅ Database migrations completed!")

if __name__ == "__main__":
    run_migrations()
