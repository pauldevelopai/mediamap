#!/usr/bin/env python3
"""
Integrate Multi-App Architecture into app.py
============================================

This script integrates the multi-app architecture into the main Flask application.
"""

import re

def update_app_py():
    """Update app.py to integrate multi-app architecture"""
    
    print("🔧 Updating app.py for multi-app architecture...")
    
    # Read the current app.py
    with open('backend/app.py', 'r') as f:
        content = f.read()
    
    # 1. Add import for app_routes at the top (after other imports)
    import_pattern = r'(from openai import OpenAI\nimport json\nimport uuid)'
    import_replacement = r'\1\n\n# Multi-app architecture\nfrom app_routes import register_app_routes'
    
    if 'from app_routes import register_app_routes' not in content:
        content = re.sub(import_pattern, import_replacement, content)
        print("✅ Added app_routes import")
    
    # 2. Register app routes after app creation (find a good spot after app initialization)
    app_init_pattern = r'(app\.config\[\'SQLALCHEMY_TRACK_MODIFICATIONS\'\] = False\n)'
    app_init_replacement = r'\1\n# Register multi-app routes\nregister_app_routes(app)\n'
    
    if 'register_app_routes(app)' not in content:
        content = re.sub(app_init_pattern, app_init_replacement, content)
        print("✅ Added app routes registration")
    
    # 3. Update the login route to redirect to app selector instead of admin dashboard
    login_route_pattern = r'(login_user\(user\)\s*\n\s*flash\([^)]+\)\s*\n\s*)return redirect\(url_for\([\'"]admin_dashboard[\'"]\)\)'
    login_route_replacement = r'\1return redirect(url_for(\'app_selector\'))'
    
    content = re.sub(login_route_pattern, login_route_replacement, content, flags=re.MULTILINE)
    print("✅ Updated login route to redirect to app selector")
    
    # 4. Update the root route to redirect to app selector if logged in
    root_route_pattern = r'@app\.route\(\'/\'\)\ndef index\(\):\s*if current_user\.is_authenticated:\s*return redirect\(url_for\([\'"]admin_dashboard[\'"]\)\)'
    root_route_replacement = '''@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('app_selector'))'''
    
    content = re.sub(root_route_pattern, root_route_replacement, content, flags=re.MULTILINE | re.DOTALL)
    print("✅ Updated root route to redirect to app selector")
    
    # Write back the updated content
    with open('backend/app.py', 'w') as f:
        f.write(content)
    
    print("✅ app.py updated successfully")

def update_admin_base_template():
    """Update admin base template to use filtered sidebars based on app context"""
    
    print("🔧 Updating admin base template...")
    
    # Read the admin base template
    try:
        with open('backend/templates/admin/base_admin.html', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("⚠️ Admin base template not found, skipping...")
        return
    
    # Find the sidebar section and replace it with conditional includes
    sidebar_pattern = r'<div class="sidebar"[^>]*>.*?</div>\s*<!-- End sidebar -->'
    
    sidebar_replacement = '''<!-- Dynamic Sidebar Based on App Context -->
        {% if is_mediamap_admin %}
            {% include 'admin/sidebars/mediamap_admin_sidebar.html' %}
        {% elif is_healthpin_admin %}
            {% include 'admin/sidebars/healthpin_admin_sidebar.html' %}
        {% else %}
            <!-- Default sidebar for unknown context -->
            <div class="sidebar" id="sidebar">
                <div class="sidebar-header">
                    <h4>Unknown App</h4>
                </div>
                <div class="sidebar-content">
                    <div class="text-center py-4">
                        <p>Please select an app from the <a href="{{ url_for('app_selector') }}">app selector</a></p>
                    </div>
                </div>
            </div>
        {% endif %}
        <!-- End sidebar -->'''
    
    if re.search(sidebar_pattern, content, re.DOTALL):
        content = re.sub(sidebar_pattern, sidebar_replacement, content, flags=re.DOTALL)
        print("✅ Updated sidebar to use conditional includes")
    else:
        print("⚠️ Could not find sidebar pattern in admin base template")
    
    # Write back the updated template
    with open('backend/templates/admin/base_admin.html', 'w') as f:
        f.write(content)
    
    print("✅ Admin base template updated")

def create_base_template_for_user_apps():
    """Create a base template for user apps (MediaMap and HealthPIN user interfaces)"""
    
    print("🔧 Creating base template for user apps...")
    
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}{{ app_name }}{% endblock %}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/bootstrap-icons.min.css" rel="stylesheet">
    
    <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .navbar-brand {
            font-weight: 700;
            font-size: 1.5rem;
        }
        
        .main-content {
            padding: 2rem 0;
        }
        
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        
        .btn {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .stats-card {
            border-radius: 15px;
            color: white;
            margin-bottom: 1rem;
        }
        
        .footer {
            background-color: #343a40;
            color: white;
            padding: 2rem 0;
            margin-top: 4rem;
        }
    </style>
    
    {% block extra_css %}{% endblock %}
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                {% if is_mediamap_user %}
                    <i class="bi bi-newspaper me-2"></i>MediaMap
                {% elif is_healthpin_user %}
                    <i class="bi bi-heart-pulse-fill me-2"></i>HealthPIN
                {% else %}
                    <i class="bi bi-grid-3x3-gap me-2"></i>{{ app_name }}
                {% endif %}
            </a>
            
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    {% if is_mediamap_user %}
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('mediamap_user_dashboard') }}">
                                <i class="bi bi-house me-1"></i>Dashboard
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('new_media_analysis') }}">
                                <i class="bi bi-graph-up me-1"></i>Analysis
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('new_content') }}">
                                <i class="bi bi-file-text me-1"></i>Content
                            </a>
                        </li>
                    {% elif is_healthpin_user %}
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('healthpin_user_dashboard') }}">
                                <i class="bi bi-house me-1"></i>Dashboard
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('find_doctor') }}">
                                <i class="bi bi-search me-1"></i>Find Doctor
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('patient_matching') }}">
                                <i class="bi bi-heart-arrow me-1"></i>Matching
                            </a>
                        </li>
                    {% endif %}
                </ul>
                
                <ul class="navbar-nav">
                    <li class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle" href="#" id="userDropdown" role="button" data-bs-toggle="dropdown">
                            <i class="bi bi-person-circle me-1"></i>{{ current_user.username }}
                        </a>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" href="{{ url_for('app_selector') }}">
                                <i class="bi bi-grid-3x3-gap me-2"></i>Switch App
                            </a></li>
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item" href="{{ url_for('logout') }}">
                                <i class="bi bi-box-arrow-right me-2"></i>Logout
                            </a></li>
                        </ul>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
    
    <!-- Main Content -->
    <main class="main-content">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="container">
                    {% for category, message in messages %}
                        <div class="alert alert-{{ 'danger' if category == 'error' else category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        </div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </main>
    
    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5>{{ app_name }}</h5>
                    <p class="mb-0">
                        {% if is_mediamap_user %}
                            Media analysis and content creation platform
                        {% elif is_healthpin_user %}
                            Healthcare data analysis and patient matching
                        {% else %}
                            Multi-application platform
                        {% endif %}
                    </p>
                </div>
                <div class="col-md-6 text-md-end">
                    <p class="mb-0">&copy; 2025 MediaMap Platform. All rights reserved.</p>
                </div>
            </div>
        </div>
    </footer>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    
    {% block extra_js %}{% endblock %}
</body>
</html>'''
    
    with open('backend/templates/base.html', 'w') as f:
        f.write(base_template)
    
    print("✅ Created base template for user apps")

def main():
    """Main function to integrate multi-app architecture"""
    
    print("🔧 INTEGRATING MULTI-APP ARCHITECTURE")
    print("=====================================")
    
    update_app_py()
    update_admin_base_template()
    create_base_template_for_user_apps()
    
    print("")
    print("✅ MULTI-APP ARCHITECTURE INTEGRATED!")
    print("=====================================")
    print("")
    print("🎯 What's been integrated:")
    print("• app.py updated with multi-app routes")
    print("• Login redirects to app selector")
    print("• Admin templates use filtered sidebars")
    print("• Base template created for user apps")
    print("")
    print("🧪 Test the flow:")
    print("1. Login → App Selector")
    print("2. Choose MediaMap → MediaMap user interface")
    print("3. Choose MediaMap Admin → Admin with MediaMap functions")
    print("4. Choose HealthPIN → HealthPIN user interface")
    print("5. Choose HealthPIN Admin → Admin with HealthPIN functions")

if __name__ == "__main__":
    main()
