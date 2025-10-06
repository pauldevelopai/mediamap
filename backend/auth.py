from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
try:
    from models import User
except ImportError:
    from models import User
try:
    from session_manager import SessionManager
except ImportError:
    from session_manager import SessionManager

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        access_level = request.form.get('access_level', 'full')
        session_notes = request.form.get('session_notes', '')
        
        # Authenticate admin user
        user = SessionManager.authenticate_admin(username, password)
        
        if user:
            # Create secure session with memory access control
            session_token = SessionManager.create_session(
                user_id=user.id,
                access_level=access_level,
                session_notes=session_notes
            )
            
            # Store user info in Flask session
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            session['access_level'] = access_level
            
            # Store initial memory about this login
            current_session = SessionManager.get_current_session()
            if current_session:
                SessionManager.store_memory(
                    session_id=current_session.id,
                    memory_type='login',
                    content=f"Admin login: {username} with {access_level} access level",
                    metadata={
                        'ip_address': request.remote_addr,
                        'user_agent': request.headers.get('User-Agent'),
                        'session_notes': session_notes
                    },
                    importance_score=0.3
                )
            
            flash('Login successful!', 'success')
            return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@auth.route('/logout')
@login_required
def logout():
    # Logout from session manager
    SessionManager.logout_session()
    
    # Clear Flask session
    session.clear()
    
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'error')
            return render_template('register.html')
        
        # Create admin user
        try:
            user = SessionManager.create_admin_user(username, password, email)
            flash('Admin account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error creating account: {str(e)}', 'error')
    
    return render_template('register.html')

@auth.route('/admin/create', methods=['GET', 'POST'])
def create_admin():
    """Create admin user endpoint"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        try:
            user = SessionManager.create_admin_user(username, password, email)
            return {'success': True, 'message': f'Admin user {username} created successfully'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    return render_template('admin/create_admin.html')
