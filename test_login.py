import sys
sys.path.append('/opt/mediamap')
from backend.app import app, db, User
from werkzeug.security import check_password_hash

with app.app_context():
    try:
        user = User.query.filter_by(username='admin').first()
        if user:
            print('Found user:', user.username)
            print('Is admin:', user.is_admin)
            print('Password hash:', user.password_hash[:50] + '...')
            if check_password_hash(user.password_hash, 'admin123'):
                print('Password check: SUCCESS')
            else:
                print('Password check: FAILED')
        else:
            print('No user found')
    except Exception as e:
        print('Error:', str(e))
