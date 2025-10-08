#!/usr/bin/env python3
"""
Fix Admin Routing
================

Fix the routing to properly direct users to the new separate admin apps
instead of the old combined admin interface.
"""

import re

def fix_app_routes_redirect():
    """Fix the app routes to properly redirect to separate admin apps"""
    
    print("🔧 Fixing app routes redirect logic...")
    
    with open('backend/app_routes.py', 'r') as f:
        content = f.read()
    
    # Find and replace the set_app_context function
    new_set_app_context = '''    @app.route('/set-app-context/<app_type>')
    @login_required
    def set_app_context(app_type):
        """Set the app context and redirect to appropriate dashboard"""
        
        # Store app context in session
        session['app_context'] = app_type
        session['app_name'] = {
            'mediamap': 'MediaMap',
            'mediamap_admin': 'MediaMap Admin',
            'healthpin': 'HealthPIN',
            'healthpin_admin': 'HealthPIN Admin'
        }.get(app_type, 'Unknown')
        
        # Redirect based on app type to NEW separate admin apps
        if app_type == 'mediamap':
            return redirect(url_for('mediamap_user_dashboard'))
        elif app_type == 'mediamap_admin':
            # Redirect to NEW MediaMap Admin app
            return redirect('/mediamap-admin/')
        elif app_type == 'healthpin':
            return redirect(url_for('healthpin_user_dashboard'))
        elif app_type == 'healthpin_admin':
            # Redirect to NEW HealthPIN Admin app
            return redirect('/healthpin-admin/')
        else:
            return redirect(url_for('app_selector'))'''
    
    # Replace the existing function
    pattern = r'@app\.route\(\'/set-app-context/<app_type>\'\).*?return redirect\(url_for\(\'app_selector\'\)\)'
    content = re.sub(pattern, new_set_app_context, content, flags=re.DOTALL)
    
    with open('backend/app_routes.py', 'w') as f:
        f.write(content)
    
    print("✅ App routes redirect logic fixed")

def fix_login_handler():
    """Fix the login handler to redirect to separate admin apps"""
    
    print("🔧 Fixing login handler...")
    
    with open('backend/app_routes.py', 'r') as f:
        content = f.read()
    
    # Find and update the handle_login_with_app_selection function
    new_login_handler = '''    def handle_login_with_app_selection(app_type):
        """Handle login with direct app selection"""
        
        # Store app context in session
        session['app_context'] = app_type
        session['app_name'] = {
            'mediamap': 'MediaMap',
            'mediamap_admin': 'MediaMap Admin',
            'healthpin': 'HealthPIN',
            'healthpin_admin': 'HealthPIN Admin'
        }.get(app_type, 'Unknown')
        
        # Redirect based on app type to NEW separate admin apps
        if app_type == 'mediamap':
            return redirect(url_for('mediamap_user_dashboard'))
        elif app_type == 'mediamap_admin':
            # Redirect to NEW MediaMap Admin app
            return redirect('/mediamap-admin/')
        elif app_type == 'healthpin':
            return redirect(url_for('healthpin_user_dashboard'))
        elif app_type == 'healthpin_admin':
            # Redirect to NEW HealthPIN Admin app
            return redirect('/healthpin-admin/')
        else:
            return redirect(url_for('app_selector'))'''
    
    # Replace the existing function
    pattern = r'def handle_login_with_app_selection\(app_type\):.*?return redirect\(url_for\(\'app_selector\'\)\)'
    content = re.sub(pattern, new_login_handler, content, flags=re.DOTALL)
    
    with open('backend/app_routes.py', 'w') as f:
        f.write(content)
    
    print("✅ Login handler fixed")

def disable_old_admin_routes():
    """Disable or redirect old admin routes to prevent confusion"""
    
    print("🔧 Disabling old admin routes...")
    
    with open('backend/app.py', 'r') as f:
        content = f.read()
    
    # Find the old admin_dashboard route and redirect it to app selector
    old_admin_pattern = r'@app\.route\(\'/admin\'\).*?def admin_dashboard\(\):.*?return render_template\([^)]+\)'
    
    new_admin_redirect = '''@app.route('/admin')
@app.route('/admin/')
@login_required
def admin_dashboard():
    """Redirect old admin route to app selector"""
    flash('Please select your admin application', 'info')
    return redirect(url_for('app_selector'))'''
    
    if re.search(old_admin_pattern, content, re.DOTALL):
        content = re.sub(old_admin_pattern, new_admin_redirect, content, flags=re.DOTALL)
        print("✅ Old admin dashboard route redirected to app selector")
    
    # Also redirect any /admin/* routes to app selector
    admin_catch_all = '''
@app.route('/admin/<path:path>')
@login_required
def admin_catch_all(path):
    """Redirect old admin paths to app selector"""
    flash('Please select your admin application from the options below', 'warning')
    return redirect(url_for('app_selector'))'''
    
    if '@app.route(\'/admin/<path:path>\')' not in content:
        # Add the catch-all route before the final if __name__ == '__main__':
        content = content.replace('if __name__ == \'__main__\':', admin_catch_all + '\n\nif __name__ == \'__main__\':')
        print("✅ Added catch-all redirect for old admin paths")
    
    with open('backend/app.py', 'w') as f:
        f.write(content)
    
    print("✅ Old admin routes disabled")

def create_test_routes():
    """Create test routes to verify the separate admin apps work"""
    
    print("🧪 Creating test routes...")
    
    test_routes = '''
# Test routes for separate admin apps
@app.route('/test-mediamap-admin')
@login_required
def test_mediamap_admin():
    """Test route for MediaMap Admin"""
    session['app_context'] = 'mediamap_admin'
    return redirect('/mediamap-admin/')

@app.route('/test-healthpin-admin')
@login_required
def test_healthpin_admin():
    """Test route for HealthPIN Admin"""
    session['app_context'] = 'healthpin_admin'
    return redirect('/healthpin-admin/')'''
    
    with open('backend/app.py', 'r') as f:
        content = f.read()
    
    if '/test-mediamap-admin' not in content:
        content = content.replace('if __name__ == \'__main__\':', test_routes + '\n\nif __name__ == \'__main__\':')
        
        with open('backend/app.py', 'w') as f:
            f.write(content)
        
        print("✅ Test routes created")

def main():
    """Main function to fix admin routing"""
    
    print("🔧 FIXING ADMIN ROUTING")
    print("======================")
    
    fix_app_routes_redirect()
    fix_login_handler()
    disable_old_admin_routes()
    create_test_routes()
    
    print("")
    print("✅ ADMIN ROUTING FIXED!")
    print("======================")
    print("")
    print("🎯 What's been fixed:")
    print("• App routes now redirect to separate admin apps")
    print("• Login handler redirects to /mediamap-admin/ and /healthpin-admin/")
    print("• Old /admin routes redirect to app selector")
    print("• Test routes created for verification")
    print("")
    print("🎯 New routing:")
    print("• HealthPIN Admin login → /healthpin-admin/")
    print("• MediaMap Admin login → /mediamap-admin/")
    print("• Old /admin/* → App selector with message")
    print("")
    print("🧪 Test URLs:")
    print("• http://localhost:8080/test-healthpin-admin")
    print("• http://localhost:8080/test-mediamap-admin")
    print("")
    print("🔄 Restart your app to apply the routing fixes!")

if __name__ == "__main__":
    main()
