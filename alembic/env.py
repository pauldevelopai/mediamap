import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

try:
    from backend.models import db
    target_metadata = db.metadata
except Exception as e:
    print(f"Alembic import error: {e}")
    target_metadata = None 