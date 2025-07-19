import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from models import db
    target_metadata = db.metadata
except Exception as e:
    print(f"Alembic import error: {e}")
    target_metadata = None 