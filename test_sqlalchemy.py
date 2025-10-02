import sys
sys.path.append('/opt/mediamap')
from backend.app import app, db, User
from werkzeug.security import check_password_hash

with app.app_context():
    try:
        user = User.query.filter_by(username='admin').first()
        if user:
            print('SUCCESS: Found admin user:', user.username)
            print('Is admin:', user.is_admin)
            if check_password_hash(user.password_hash, 'admin123'):
                print('SUCCESS: Password check passed')
            else:
                print('ERROR: Password check failed')
        else:
            print('ERROR: No admin user found')
    except Exception as e:
        print('ERROR:', str(e))
