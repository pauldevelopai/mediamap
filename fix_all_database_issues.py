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
        # First, let's just create the organisations table specifically
        from backend.aimap.models import Organisation
        
        print("Creating organisations table...")
        Organisation.__table__.create(db.engine, checkfirst=True)
        print("Organisations table created successfully!")
        
        # Verify it exists
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        if 'organisations' in tables:
            print("SUCCESS: organisations table verified!")
            
            # Test a simple query
            count = Organisation.query.count()
            print(f"Current organisations count: {count}")
        else:
            print("ERROR: organisations table not found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
