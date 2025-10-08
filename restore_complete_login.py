#!/usr/bin/env python3
"""
Restore Complete Login with All 4 Apps
======================================

Restore the login screen to show all 4 original apps:
- MediaMap (user)
- MediaMap Admin 
- HealthPIN (user)
- HealthPIN Admin
"""

import re

def restore_login_template():
    """Restore the complete login template with all 4 apps"""
    
    print("🔧 Restoring complete login template...")
    
    # Read current login template
    with open('backend/templates/login.html', 'r') as f:
        content = f.read()
    
    # Ensure the app selector dropdown has all 4 options
    complete_app_selector = '''                    <div class="mb-3">
                        <label for="app_type" class="form-label">Select Application</label>
                        <div class="input-group">
                            <span class="input-group-text"><i class="bi bi-grid-3x3-gap"></i></span>
                            <select id="app_type" name="app_type" class="form-select" required>
                                <option value="">Choose your application...</option>
                                <option value="mediamap">📰 MediaMap - Media Analysis & Content (User)</option>
                                <option value="mediamap_admin">⚙️ MediaMap Admin - Media Management (Admin)</option>
                                <option value="healthpin">🏥 HealthPIN - Healthcare Data & Matching (User)</option>
                                <option value="healthpin_admin">🏥 HealthPIN Admin - Healthcare Management (Admin)</option>
                            </select>
                        </div>
                    </div>'''
    
    # Replace any existing app selector with the complete one
    app_selector_pattern = r'<div class="mb-3">\s*<label for="app_type".*?</div>\s*</div>'
    
    if re.search(app_selector_pattern, content, re.DOTALL):
        content = re.sub(app_selector_pattern, complete_app_selector, content, flags=re.DOTALL)
        print("✅ Updated existing app selector with all 4 apps")
    else:
        # If no app selector exists, add it before the username field
        username_pattern = r'(<div class="mb-3">\s*<label for="username")'
        content = re.sub(username_pattern, complete_app_selector + '\n                    \n                    \\1', content)
        print("✅ Added complete app selector with all 4 apps")
    
    # Write back the updated template
    with open('backend/templates/login.html', 'w') as f:
        f.write(content)
    
    print("✅ Login template restored with all 4 apps")

def fix_app_routes_for_all_apps():
    """Fix app routes to handle all 4 app types"""
    
    print("🔧 Fixing app routes for all 4 apps...")
    
    with open('backend/app_routes.py', 'r') as f:
        content = f.read()
    
    # Fix the handle_login_with_app_selection function
    complete_login_handler = '''    def handle_login_with_app_selection(app_type):
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
            return redirect(url_for('app_selector'))'''
    
    # Replace the existing function
    pattern = r'def handle_login_with_app_selection\(app_type\):.*?return redirect\(url_for\(\'app_selector\'\)\)'
    content = re.sub(pattern, complete_login_handler, content, flags=re.DOTALL)
    
    # Also fix the set_app_context function
    complete_set_context = '''    @app.route('/set-app-context/<app_type>')
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
            return redirect(url_for('app_selector'))'''
    
    # Replace the set_app_context function
    set_context_pattern = r'@app\.route\(\'/set-app-context/<app_type>\'\).*?return redirect\(url_for\(\'app_selector\'\)\)'
    content = re.sub(set_context_pattern, complete_set_context, content, flags=re.DOTALL)
    
    with open('backend/app_routes.py', 'w') as f:
        f.write(content)
    
    print("✅ App routes fixed for all 4 apps")

def update_app_selector_template():
    """Update app selector template to show all 4 apps"""
    
    print("🔧 Updating app selector template...")
    
    with open('backend/templates/app_selector.html', 'r') as f:
        content = f.read()
    
    # Ensure all 4 app cards are present
    complete_app_cards = '''        <div class="row g-4">
            <!-- MediaMap User App -->
            <div class="col-md-6 col-lg-3">
                <a href="{{ url_for('set_app_context', app_type='mediamap') }}" class="app-card mediamap-card d-block">
                    <i class="bi bi-newspaper app-icon"></i>
                    <h3 class="app-title">MediaMap</h3>
                    <p class="app-description">Media analysis and content creation tools for users</p>
                    <ul class="app-features">
                        <li><i class="bi bi-check-circle-fill"></i> Media Analysis</li>
                        <li><i class="bi bi-check-circle-fill"></i> Content Creation</li>
                        <li><i class="bi bi-check-circle-fill"></i> AI Insights</li>
                        <li><i class="bi bi-check-circle-fill"></i> Reports & Analytics</li>
                    </ul>
                </a>
            </div>
            
            <!-- MediaMap Admin -->
            <div class="col-md-6 col-lg-3">
                <a href="{{ url_for('set_app_context', app_type='mediamap_admin') }}" class="app-card mediamap-admin-card d-block">
                    <i class="bi bi-gear-fill app-icon"></i>
                    <h3 class="app-title">MediaMap Admin</h3>
                    <p class="app-description">Administrative interface for MediaMap management</p>
                    <ul class="app-features">
                        <li><i class="bi bi-check-circle-fill"></i> User Management</li>
                        <li><i class="bi bi-check-circle-fill"></i> Media Config</li>
                        <li><i class="bi bi-check-circle-fill"></i> AI Agents</li>
                        <li><i class="bi bi-check-circle-fill"></i> System Settings</li>
                    </ul>
                </a>
            </div>
            
            <!-- HealthPIN User App -->
            <div class="col-md-6 col-lg-3">
                <a href="{{ url_for('set_app_context', app_type='healthpin') }}" class="app-card healthpin-card d-block">
                    <i class="bi bi-heart-pulse-fill app-icon"></i>
                    <h3 class="app-title">HealthPIN</h3>
                    <p class="app-description">Healthcare data analysis and patient matching</p>
                    <ul class="app-features">
                        <li><i class="bi bi-check-circle-fill"></i> Patient Data</li>
                        <li><i class="bi bi-check-circle-fill"></i> Doctor Matching</li>
                        <li><i class="bi bi-check-circle-fill"></i> Health Insights</li>
                        <li><i class="bi bi-check-circle-fill"></i> Medical Records</li>
                    </ul>
                </a>
            </div>
            
            <!-- HealthPIN Admin -->
            <div class="col-md-6 col-lg-3">
                <a href="{{ url_for('set_app_context', app_type='healthpin_admin') }}" class="app-card healthpin-admin-card d-block">
                    <i class="bi bi-hospital-fill app-icon"></i>
                    <h3 class="app-title">HealthPIN Admin</h3>
                    <p class="app-description">Administrative interface for HealthPIN management</p>
                    <ul class="app-features">
                        <li><i class="bi bi-check-circle-fill"></i> Healthcare Config</li>
                        <li><i class="bi bi-check-circle-fill"></i> Doctor Management</li>
                        <li><i class="bi bi-check-circle-fill"></i> Health Agents</li>
                        <li><i class="bi bi-check-circle-fill"></i> Medical Analytics</li>
                    </ul>
                </a>
            </div>
        </div>'''
    
    # Replace the app cards section
    cards_pattern = r'<div class="row g-4">.*?</div>'
    content = re.sub(cards_pattern, complete_app_cards, content, flags=re.DOTALL)
    
    with open('backend/templates/app_selector.html', 'w') as f:
        f.write(content)
    
    print("✅ App selector template updated with all 4 apps")

def fix_indentation_error():
    """Fix the indentation error in app_routes.py"""
    
    print("🔧 Fixing indentation error in app_routes.py...")
    
    with open('backend/app_routes.py', 'r') as f:
        content = f.read()
    
    # Fix any indentation issues
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix common indentation issues
        if line.strip().startswith('@login_required') and i > 0:
            # Ensure proper indentation for decorators
            if not lines[i-1].strip().endswith(':'):
                fixed_lines.append('    ' + line.strip())
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    fixed_content = '\n'.join(fixed_lines)
    
    with open('backend/app_routes.py', 'w') as f:
        f.write(fixed_content)
    
    print("✅ Fixed indentation errors")

def main():
    """Main function to restore complete login"""
    
    print("🔄 RESTORING COMPLETE LOGIN WITH ALL 4 APPS")
    print("===========================================")
    
    restore_login_template()
    fix_app_routes_for_all_apps()
    update_app_selector_template()
    fix_indentation_error()
    
    print("")
    print("✅ COMPLETE LOGIN RESTORED!")
    print("===========================")
    print("")
    print("🎯 All 4 apps now available in login:")
    print("• 📰 MediaMap - Media Analysis & Content (User)")
    print("• ⚙️ MediaMap Admin - Media Management (Admin)")
    print("• 🏥 HealthPIN - Healthcare Data & Matching (User)")
    print("• 🏥 HealthPIN Admin - Healthcare Management (Admin)")
    print("")
    print("🎯 Login flow:")
    print("1. User sees dropdown with all 4 options")
    print("2. Selects their desired app/role")
    print("3. Gets directed to appropriate interface:")
    print("   - User apps: Regular user interfaces")
    print("   - Admin apps: Separate dedicated admin interfaces")
    print("")
    print("🔄 Restart your app to see the complete login!")

if __name__ == "__main__":
    main()
