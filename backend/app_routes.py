"""
App Context Management Routes
============================

Routes for handling multi-app architecture and role-based access control.
"""

from flask import session, redirect, url_for, render_template, request, jsonify
from flask_login import login_required, current_user

def register_app_routes(app):
    """Register app context management routes"""
    
    @app.route('/app-selector')
    @login_required
    def app_selector():
        """Show app selector page after login"""
        return render_template('app_selector.html')
    
            @app.route('/set-app-context/<app_type>')
    @login_required
    def set_app_context(app_type):
        """Set the app context and redirect to appropriate dashboard - ALL 4 APPS"""
        
        # Store app context in session
        session['app_context'] = app_type
        session['app_name'] = {
            'mediamap': 'MediaMap',
            'mediamap_admin': 'MediaMap Admin',
            'healthpin': 'HealthPIN',
            'healthpin_admin': 'HealthPIN Admin'
        }.get(app_type, 'Unknown')
        
        # Redirect based on app type - ALL 4 OPTIONS
        if app_type == 'mediamap':
            # MediaMap USER interface
            return redirect(url_for('mediamap_user_dashboard'))
        elif app_type == 'mediamap_admin':
            # MediaMap ADMIN interface (separate app)
            return redirect('/mediamap-admin/')
        elif app_type == 'healthpin':
            # HealthPIN USER interface
            return redirect(url_for('healthpin_user_dashboard'))
        elif app_type == 'healthpin_admin':
            # HealthPIN ADMIN interface (separate app)
            return redirect('/healthpin-admin/')
        else:
            return redirect(url_for('app_selector'))
    
    @app.route('/mediamap-dashboard')
    @login_required
    def mediamap_user_dashboard():
        """MediaMap user dashboard"""
        if session.get('app_context') != 'mediamap':
            return redirect(url_for('app_selector'))
        
        return render_template('mediamap/user_dashboard.html')
    
    @app.route('/healthpin-user-dashboard')
    @login_required
    def healthpin_user_dashboard():
        """HealthPIN user dashboard"""
        if session.get('app_context') != 'healthpin':
            return redirect(url_for('app_selector'))
        
        return render_template('healthpin/user_dashboard.html')
    
    @app.context_processor
    def inject_app_context():
        """Inject app context into all templates"""
        return {
            'app_context': session.get('app_context', 'unknown'),
            'app_name': session.get('app_name', 'Unknown App'),
            'is_mediamap_admin': session.get('app_context') == 'mediamap_admin',
            'is_healthpin_admin': session.get('app_context') == 'healthpin_admin',
            'is_mediamap_user': session.get('app_context') == 'mediamap',
            'is_healthpin_user': session.get('app_context') == 'healthpin'
        }
    
    def require_app_context(required_context):
        """Decorator to require specific app context"""
        def decorator(f):
            def decorated_function(*args, **kwargs):
                if session.get('app_context') != required_context:
                    return redirect(url_for('app_selector'))
                return f(*args, **kwargs)
            decorated_function.__name__ = f.__name__
            return decorated_function
        return decorator
    
    # Make decorator available globally
    app.require_app_context = require_app_context
    
    
            def handle_login_with_app_selection(app_type):
        """Handle login with direct app selection for ALL 4 apps"""
        
        # Store app context in session
        session['app_context'] = app_type
        session['app_name'] = {
            'mediamap': 'MediaMap',
            'mediamap_admin': 'MediaMap Admin',
            'healthpin': 'HealthPIN',
            'healthpin_admin': 'HealthPIN Admin'
        }.get(app_type, 'Unknown')
        
        # Redirect based on app type - ALL 4 OPTIONS
        if app_type == 'mediamap':
            # MediaMap USER interface
            return redirect(url_for('mediamap_user_dashboard'))
        elif app_type == 'mediamap_admin':
            # MediaMap ADMIN interface (separate app)
            return redirect('/mediamap-admin/')
        elif app_type == 'healthpin':
            # HealthPIN USER interface  
            return redirect(url_for('healthpin_user_dashboard'))
        elif app_type == 'healthpin_admin':
            # HealthPIN ADMIN interface (separate app)
            return redirect('/healthpin-admin/')
        else:
            return redirect(url_for('app_selector'))
    
    # Make this function available to the main app
    app.handle_login_with_app_selection = handle_login_with_app_selection

    print("✅ Registered app context management routes")
