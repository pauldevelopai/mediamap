import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import db, User
from werkzeug.security import generate_password_hash
from app import app

with app.app_context():
    if not User.query.filter_by(username='paul').first():
        u = User(username='paul', email='paul@example.com', password_hash=generate_password_hash('password'))
        db.session.add(u)
        db.session.commit()
        print('User created: paul')
    else:
        print('User already exists: paul') 