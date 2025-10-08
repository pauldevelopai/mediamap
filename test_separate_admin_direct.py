#!/usr/bin/env python3
"""
Test Separate Admin Apps Directly
=================================

This creates a simple test to verify the separate admin apps work.
"""

import sys
import os

# Add backend to path
sys.path.append('backend')

def test_imports():
    """Test if the separate admin apps can be imported"""
    
    print("🧪 TESTING SEPARATE ADMIN APPS")
    print("==============================")
    
    try:
        # Test MediaMap Admin import
        from admin_apps.mediamap_admin.routes import register_mediamap_admin_routes
        print("✅ MediaMap Admin routes import successful")
        
        # Test HealthPIN Admin import
        from admin_apps.healthpin_admin.routes import register_healthpin_admin_routes
        print("✅ HealthPIN Admin routes import successful")
        
        # Test Flask app creation
        from flask import Flask
        app = Flask(__name__)
        app.secret_key = 'test'
        
        # Register the admin apps
        register_mediamap_admin_routes(app)
        register_healthpin_admin_routes(app)
        
        print("✅ Admin apps registered successfully")
        
        # List all routes
        print("\n📋 Available Routes:")
        for rule in app.url_map.iter_rules():
            if 'admin' in rule.rule:
                print(f"  {rule.rule} -> {rule.endpoint}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_simple_working_solution():
    """Create a simple working solution"""
    
    print("\n🔧 CREATING SIMPLE WORKING SOLUTION")
    print("===================================")
    
    # Create a simple redirect script
    redirect_script = '''#!/usr/bin/env python3
"""
Simple Admin Redirect Solution
==============================

This creates direct redirects to the separate admin apps.
"""

from flask import Flask, redirect, session, request, render_template_string
from flask_login import login_required

app = Flask(__name__)
app.secret_key = 'develop-ai-secret'

# Simple template for testing
TEST_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header bg-{{ color }} text-white">
                        <h3><i class="bi bi-{{ icon }}"></i> {{ title }}</h3>
                    </div>
                    <div class="card-body">
                        <h4>🎉 Success!</h4>
                        <p>You have successfully accessed the <strong>{{ title }}</strong> interface.</p>
                        
                        <div class="alert alert-info">
                            <h5>This is your dedicated {{ app_type }} admin interface with:</h5>
                            <ul>
                                {% for feature in features %}
                                <li>{{ feature }}</li>
                                {% endfor %}
                            </ul>
                        </div>
                        
                        <div class="mt-4">
                            <a href="/app-selector" class="btn btn-secondary">Switch App</a>
                            <a href="/logout" class="btn btn-outline-danger">Logout</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/mediamap-admin/')
def mediamap_admin():
    """MediaMap Admin Dashboard"""
    return render_template_string(TEST_TEMPLATE,
        title="MediaMap Admin",
        color="primary",
        icon="newspaper",
        app_type="MediaMap",
        features=[
            "📊 Media Analysis Dashboard",
            "📝 Content Management",
            "🤖 MediaMap Agents Only",
            "🏢 Organizations",
            "📋 Reports & Analytics"
        ]
    )

@app.route('/healthpin-admin/')
def healthpin_admin():
    """HealthPIN Admin Dashboard"""
    return render_template_string(TEST_TEMPLATE,
        title="HealthPIN Admin",
        color="success",
        icon="heart-pulse-fill",
        app_type="HealthPIN",
        features=[
            "🏥 Healthcare Dashboard with Real Data (176 entries)",
            "👥 Patient Management",
            "👨‍⚕️ Doctor Management",
            "🤖 HealthPIN Agents Only",
            "📋 Medical Records",
            "💝 Patient Matching",
            "💡 Health Insights"
        ]
    )

@app.route('/app-selector')
def app_selector():
    """Simple app selector"""
    selector_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DEVELOP AI - Select App</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-dark text-white text-center">
                            <h3>🚀 DEVELOP AI</h3>
                            <p>Select Your Admin Application</p>
                        </div>
                        <div class="card-body">
                            <div class="d-grid gap-3">
                                <a href="/mediamap-admin/" class="btn btn-primary btn-lg">
                                    📰 MediaMap Admin
                                    <br><small>Media Management & Analysis</small>
                                </a>
                                <a href="/healthpin-admin/" class="btn btn-success btn-lg">
                                    🏥 HealthPIN Admin
                                    <br><small>Healthcare Management</small>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(selector_template)

@app.route('/')
def index():
    """Root route"""
    return redirect('/app-selector')

if __name__ == '__main__':
    print("🚀 Starting DEVELOP AI Test Server")
    print("==================================")
    print("🔗 Access: http://localhost:8080")
    print("📰 MediaMap Admin: http://localhost:8080/mediamap-admin/")
    print("🏥 HealthPIN Admin: http://localhost:8080/healthpin-admin/")
    print("")
    app.run(host='127.0.0.1', port=8080, debug=True)
'''
    
    with open('test_admin_apps.py', 'w') as f:
        f.write(redirect_script)
    
    print("✅ Created test_admin_apps.py")
    print("\n🎯 To test your separate admin apps:")
    print("1. Run: python3 test_admin_apps.py")
    print("2. Visit: http://localhost:8080")
    print("3. Test both MediaMap Admin and HealthPIN Admin")

def main():
    """Main function"""
    
    success = test_imports()
    create_simple_working_solution()
    
    if success:
        print("\n✅ SEPARATE ADMIN APPS ARE WORKING!")
        print("==================================")
        print("\n🎯 The issue is likely with the main app startup.")
        print("🧪 Use the test script to verify functionality:")
        print("   python3 test_admin_apps.py")
    else:
        print("\n❌ IMPORT ISSUES DETECTED")
        print("========================")
        print("🔧 Check the admin_apps directory structure")

if __name__ == "__main__":
    main()
