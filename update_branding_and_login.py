#!/usr/bin/env python3
"""
Update Branding and Login Page
==============================

Updates the app branding to "DEVELOP AI" and adds the 4-app dropdown selector to login.
"""

import re

def update_login_template():
    """Update login template with DEVELOP AI branding and 4-app selector"""
    
    print("🔧 Updating login template...")
    
    with open('backend/templates/login.html', 'r') as f:
        content = f.read()
    
    # 1. Update branding from "MediaMap Platform" to "DEVELOP AI"
    content = content.replace('MediaMap Platform', 'DEVELOP AI')
    content = content.replace('AIMAP', 'DEVELOP AI')
    content = content.replace('Access your MediaMap and HealthPIN applications', 'Choose your AI application and get started')
    
    # 2. Add the 4-app dropdown selector back to the login form
    # Find the username field and add the selector before it
    username_field_pattern = r'(<div class="mb-3">\s*<label for="username")'
    
    app_selector_html = '''                    <div class="mb-3">
                        <label for="app_type" class="form-label">Select Application</label>
                        <div class="input-group">
                            <span class="input-group-text"><i class="bi bi-grid-3x3-gap"></i></span>
                            <select id="app_type" name="app_type" class="form-select" required>
                                <option value="">Choose your application...</option>
                                <option value="mediamap">📰 MediaMap - Media Analysis & Content</option>
                                <option value="mediamap_admin">⚙️ MediaMap Admin - Media Management</option>
                                <option value="healthpin">🏥 HealthPIN - Healthcare Data & Matching</option>
                                <option value="healthpin_admin">🏥 HealthPIN Admin - Healthcare Management</option>
                            </select>
                        </div>
                    </div>
                    
                    \\1'''
    
    content = re.sub(username_field_pattern, app_selector_html, content)
    
    # 3. Update the logo icon to something more generic for "DEVELOP AI"
    content = content.replace('bi-robot', 'bi-cpu-fill')
    
    # Write back the updated template
    with open('backend/templates/login.html', 'w') as f:
        f.write(content)
    
    print("✅ Login template updated with DEVELOP AI branding and app selector")

def update_app_selector_template():
    """Update app selector template with DEVELOP AI branding"""
    
    print("🔧 Updating app selector template...")
    
    with open('backend/templates/app_selector.html', 'r') as f:
        content = f.read()
    
    # Update branding
    content = content.replace('MediaMap Platform', 'DEVELOP AI')
    content = content.replace('Choose your application to get started', 'Select your AI-powered application')
    
    # Update the header description
    content = content.replace(
        'Multi-application platform',
        'AI-powered development and analysis platform'
    )
    
    # Write back
    with open('backend/templates/app_selector.html', 'w') as f:
        f.write(content)
    
    print("✅ App selector template updated")

def update_base_template():
    """Update base template with DEVELOP AI branding"""
    
    print("🔧 Updating base template...")
    
    with open('backend/templates/base.html', 'r') as f:
        content = f.read()
    
    # Update branding
    content = content.replace('MediaMap Platform', 'DEVELOP AI')
    content = content.replace('{{ app_name }}', 'DEVELOP AI - {{ app_name }}')
    
    # Update footer
    content = content.replace(
        '&copy; 2025 MediaMap Platform. All rights reserved.',
        '&copy; 2025 DEVELOP AI. All rights reserved.'
    )
    
    # Write back
    with open('backend/templates/base.html', 'w') as f:
        f.write(content)
    
    print("✅ Base template updated")

def update_app_routes():
    """Update app routes to handle the login app selection"""
    
    print("🔧 Updating app routes...")
    
    with open('backend/app_routes.py', 'r') as f:
        content = f.read()
    
    # Add a function to handle login with app selection
    login_handler = '''
    def handle_login_with_app_selection(app_type):
        """Handle login with direct app selection"""
        
        # Store app context in session
        session['app_context'] = app_type
        session['app_name'] = {
            'mediamap': 'MediaMap',
            'mediamap_admin': 'MediaMap Admin',
            'healthpin': 'HealthPIN',
            'healthpin_admin': 'HealthPIN Admin'
        }.get(app_type, 'Unknown')
        
        # Redirect based on app type
        if app_type == 'mediamap':
            return redirect(url_for('mediamap_user_dashboard'))
        elif app_type == 'mediamap_admin':
            return redirect(url_for('admin_dashboard'))
        elif app_type == 'healthpin':
            return redirect(url_for('healthpin_user_dashboard'))
        elif app_type == 'healthpin_admin':
            return redirect(url_for('healthpin_dashboard'))
        else:
            return redirect(url_for('app_selector'))
    
    # Make this function available to the main app
    app.handle_login_with_app_selection = handle_login_with_app_selection
'''
    
    # Add the function before the final print statement
    content = content.replace(
        'print("✅ Registered app context management routes")',
        login_handler + '\n    print("✅ Registered app context management routes")'
    )
    
    # Write back
    with open('backend/app_routes.py', 'w') as f:
        f.write(content)
    
    print("✅ App routes updated")

def update_main_app_login_route():
    """Update the main app.py login route to handle app selection"""
    
    print("🔧 Updating main app login route...")
    
    with open('backend/app.py', 'r') as f:
        content = f.read()
    
    # Find the login route and update it to handle app_type
    login_route_pattern = r'(@app\.route\(\'/login\'.*?def login\(\):.*?if user and check_password_hash\(user\.password, password\):.*?login_user\(user\).*?flash\([^)]+\).*?)return redirect\(url_for\([\'"]app_selector[\'"]\)\)'
    
    login_route_replacement = r'''\1# Check if app_type was provided in login form
        app_type = request.form.get('app_type')
        if app_type:
            # Direct login with app selection
            return app.handle_login_with_app_selection(app_type)
        else:
            # Regular login, go to app selector
            return redirect(url_for('app_selector'))'''
    
    content = re.sub(login_route_pattern, login_route_replacement, content, flags=re.DOTALL)
    
    # Write back
    with open('backend/app.py', 'w') as f:
        f.write(content)
    
    print("✅ Main app login route updated")

def update_sidebar_templates():
    """Update sidebar templates with DEVELOP AI branding"""
    
    print("🔧 Updating sidebar templates...")
    
    # MediaMap Admin Sidebar
    try:
        with open('backend/templates/admin/sidebars/mediamap_admin_sidebar.html', 'r') as f:
            content = f.read()
        
        content = content.replace('MediaMap Admin', 'DEVELOP AI - MediaMap Admin')
        
        with open('backend/templates/admin/sidebars/mediamap_admin_sidebar.html', 'w') as f:
            f.write(content)
        
        print("✅ MediaMap admin sidebar updated")
    except FileNotFoundError:
        print("⚠️ MediaMap admin sidebar not found")
    
    # HealthPIN Admin Sidebar
    try:
        with open('backend/templates/admin/sidebars/healthpin_admin_sidebar.html', 'r') as f:
            content = f.read()
        
        content = content.replace('HealthPIN Admin', 'DEVELOP AI - HealthPIN Admin')
        
        with open('backend/templates/admin/sidebars/healthpin_admin_sidebar.html', 'w') as f:
            f.write(content)
        
        print("✅ HealthPIN admin sidebar updated")
    except FileNotFoundError:
        print("⚠️ HealthPIN admin sidebar not found")

def main():
    """Main function to update branding and login"""
    
    print("🎨 UPDATING TO DEVELOP AI BRANDING")
    print("==================================")
    
    update_login_template()
    update_app_selector_template()
    update_base_template()
    update_app_routes()
    update_main_app_login_route()
    update_sidebar_templates()
    
    print("")
    print("✅ DEVELOP AI BRANDING UPDATE COMPLETE!")
    print("======================================")
    print("")
    print("🎯 What's been updated:")
    print("• Login page: DEVELOP AI branding + 4-app dropdown selector")
    print("• App selector: DEVELOP AI branding")
    print("• Base template: DEVELOP AI branding")
    print("• Sidebar templates: DEVELOP AI branding")
    print("• Login flow: Direct app selection from login page")
    print("")
    print("🎯 New Login Flow:")
    print("1. User sees DEVELOP AI login page")
    print("2. User selects app from dropdown:")
    print("   • 📰 MediaMap - Media Analysis & Content")
    print("   • ⚙️ MediaMap Admin - Media Management")
    print("   • 🏥 HealthPIN - Healthcare Data & Matching")
    print("   • 🏥 HealthPIN Admin - Healthcare Management")
    print("3. User enters credentials and logs in directly to selected app")
    print("")
    print("🔄 Restart your Flask app to see the changes!")

if __name__ == "__main__":
    main()
