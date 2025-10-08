#!/usr/bin/env python3
"""
Create Multi-App Architecture for MediaMap
==========================================

This script creates a role-based multi-app system where users can choose between:
1. MediaMap (user interface)
2. MediaMap Admin (admin interface with MediaMap functions)
3. HealthPIN (user interface)
4. HealthPIN Admin (admin interface with HealthPIN functions)
"""

import os
import shutil

def create_app_selector_template():
    """Create the app selector page shown after login"""
    
    app_selector_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Select Your App - MediaMap Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/bootstrap-icons.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 2rem 0;
        }
        
        .app-selector-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .header-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .app-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            height: 100%;
            text-decoration: none;
            color: inherit;
        }
        
        .app-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
            text-decoration: none;
            color: inherit;
        }
        
        .app-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            display: block;
        }
        
        .mediamap-card .app-icon { color: #667eea; }
        .mediamap-admin-card .app-icon { color: #764ba2; }
        .healthpin-card .app-icon { color: #28a745; }
        .healthpin-admin-card .app-icon { color: #dc3545; }
        
        .app-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .app-description {
            color: #6c757d;
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }
        
        .app-features {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .app-features li {
            padding: 0.25rem 0;
            font-size: 0.9rem;
            color: #495057;
        }
        
        .app-features li i {
            color: #28a745;
            margin-right: 0.5rem;
        }
        
        .user-info {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            color: white;
        }
        
        .logout-btn {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            text-decoration: none;
            transition: all 0.3s ease;
        }
        
        .logout-btn:hover {
            background: rgba(255, 255, 255, 0.3);
            color: white;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <a href="{{ url_for('logout') }}" class="logout-btn">
        <i class="bi bi-box-arrow-right me-2"></i>Logout
    </a>
    
    <div class="app-selector-container">
        <div class="header-card">
            <div class="d-flex align-items-center justify-content-center mb-3">
                <i class="bi bi-robot" style="font-size: 3rem; color: #667eea; margin-right: 1rem;"></i>
                <h1 class="mb-0" style="color: #333; font-weight: 700;">MediaMap Platform</h1>
            </div>
            
            <div class="user-info">
                <h5 class="mb-1">Welcome, {{ current_user.username }}!</h5>
                <p class="mb-0">Choose your application to get started</p>
            </div>
        </div>
        
        <div class="row g-4">
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
        </div>
        
        <div class="text-center mt-4">
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">
                <i class="bi bi-info-circle me-2"></i>
                Select the application that matches your role and requirements
            </p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Add smooth animations
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.app-card');
            cards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                
                setTimeout(() => {
                    card.style.transition = 'all 0.6s ease';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, 100 * (index + 1));
            });
        });
    </script>
</body>
</html>'''
    
    # Create the template
    template_path = 'backend/templates/app_selector.html'
    with open(template_path, 'w') as f:
        f.write(app_selector_html)
    
    print(f"✅ Created app selector template: {template_path}")

def create_filtered_sidebar_templates():
    """Create filtered sidebar templates for each app type"""
    
    # MediaMap Admin Sidebar (filtered)
    mediamap_admin_sidebar = '''<!-- MediaMap Admin Sidebar -->
<div class="sidebar" id="sidebar">
    <div class="sidebar-header">
        <div class="d-flex align-items-center">
            <i class="bi bi-newspaper me-2" style="font-size: 1.5rem; color: #667eea;"></i>
            <h4 class="mb-0">MediaMap Admin</h4>
        </div>
        <button class="btn btn-link sidebar-toggle d-lg-none" onclick="toggleSidebar()">
            <i class="bi bi-x-lg"></i>
        </button>
    </div>
    
    <div class="sidebar-content">
        <div class="user-info">
            <div class="d-flex align-items-center">
                <div class="user-avatar">
                    <i class="bi bi-person-circle"></i>
                </div>
                <div class="user-details">
                    <div class="user-name">{{ current_user.username }}</div>
                    <div class="user-role">MediaMap Administrator</div>
                </div>
            </div>
        </div>
        
        <nav class="sidebar-nav">
            <ul class="nav flex-column">
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_dashboard') }}">
                        <i class="bi bi-speedometer2"></i>
                        <span>Dashboard</span>
                    </a>
                </li>
                
                <!-- MediaMap Specific Functions -->
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_media_analysis') }}">
                        <i class="bi bi-graph-up"></i>
                        <span>Media Analysis</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_content') }}">
                        <i class="bi bi-file-text"></i>
                        <span>Content Management</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_agents') }}">
                        <i class="bi bi-robot"></i>
                        <span>MediaMap Agents</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_organizations') }}">
                        <i class="bi bi-building"></i>
                        <span>Organizations</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_reports') }}">
                        <i class="bi bi-file-earmark-text"></i>
                        <span>Reports</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_users') }}">
                        <i class="bi bi-people"></i>
                        <span>User Management</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_settings') }}">
                        <i class="bi bi-gear"></i>
                        <span>Settings</span>
                    </a>
                </li>
            </ul>
        </nav>
        
        <div class="sidebar-footer">
            <div class="app-switcher">
                <a href="{{ url_for('app_selector') }}" class="btn btn-outline-light btn-sm w-100">
                    <i class="bi bi-grid-3x3-gap me-2"></i>Switch App
                </a>
            </div>
            <div class="logout-section">
                <a href="{{ url_for('logout') }}" class="btn btn-outline-danger btn-sm w-100 mt-2">
                    <i class="bi bi-box-arrow-right me-2"></i>Logout
                </a>
            </div>
        </div>
    </div>
</div>'''
    
    # HealthPIN Admin Sidebar (filtered)
    healthpin_admin_sidebar = '''<!-- HealthPIN Admin Sidebar -->
<div class="sidebar" id="sidebar">
    <div class="sidebar-header">
        <div class="d-flex align-items-center">
            <i class="bi bi-heart-pulse-fill me-2" style="font-size: 1.5rem; color: #28a745;"></i>
            <h4 class="mb-0">HealthPIN Admin</h4>
        </div>
        <button class="btn btn-link sidebar-toggle d-lg-none" onclick="toggleSidebar()">
            <i class="bi bi-x-lg"></i>
        </button>
    </div>
    
    <div class="sidebar-content">
        <div class="user-info">
            <div class="d-flex align-items-center">
                <div class="user-avatar">
                    <i class="bi bi-person-circle"></i>
                </div>
                <div class="user-details">
                    <div class="user-name">{{ current_user.username }}</div>
                    <div class="user-role">HealthPIN Administrator</div>
                </div>
            </div>
        </div>
        
        <nav class="sidebar-nav">
            <ul class="nav flex-column">
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('healthpin_dashboard') }}">
                        <i class="bi bi-speedometer2"></i>
                        <span>HealthPIN Dashboard</span>
                    </a>
                </li>
                
                <!-- HealthPIN Specific Functions -->
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('healthpin_patients') }}">
                        <i class="bi bi-person-heart"></i>
                        <span>Patient Management</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('healthpin_doctors') }}">
                        <i class="bi bi-person-badge"></i>
                        <span>Doctor Management</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_agents') }}">
                        <i class="bi bi-robot"></i>
                        <span>HealthPIN Agents</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('healthpin_records') }}">
                        <i class="bi bi-file-medical"></i>
                        <span>Medical Records</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('healthpin_matches') }}">
                        <i class="bi bi-heart-arrow"></i>
                        <span>Patient Matching</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_insights') }}">
                        <i class="bi bi-lightbulb"></i>
                        <span>Health Insights</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_users') }}">
                        <i class="bi bi-people"></i>
                        <span>User Management</span>
                    </a>
                </li>
                
                <li class="nav-item">
                    <a class="nav-link" href="{{ url_for('admin_settings') }}">
                        <i class="bi bi-gear"></i>
                        <span>Settings</span>
                    </a>
                </li>
            </ul>
        </nav>
        
        <div class="sidebar-footer">
            <div class="app-switcher">
                <a href="{{ url_for('app_selector') }}" class="btn btn-outline-light btn-sm w-100">
                    <i class="bi bi-grid-3x3-gap me-2"></i>Switch App
                </a>
            </div>
            <div class="logout-section">
                <a href="{{ url_for('logout') }}" class="btn btn-outline-danger btn-sm w-100 mt-2">
                    <i class="bi bi-box-arrow-right me-2"></i>Logout
                </a>
            </div>
        </div>
    </div>
</div>'''
    
    # Create the sidebar templates
    os.makedirs('backend/templates/admin/sidebars', exist_ok=True)
    
    with open('backend/templates/admin/sidebars/mediamap_admin_sidebar.html', 'w') as f:
        f.write(mediamap_admin_sidebar)
    
    with open('backend/templates/admin/sidebars/healthpin_admin_sidebar.html', 'w') as f:
        f.write(healthpin_admin_sidebar)
    
    print("✅ Created filtered sidebar templates")

def update_login_template():
    """Update the login template to remove section selector (will be handled after login)"""
    
    # Read current login template
    with open('backend/templates/login.html', 'r') as f:
        content = f.read()
    
    # Remove the section selector from login (we'll handle app selection after login)
    updated_content = content.replace('''                    <div class="mb-3">
                        <label for="section" class="form-label">Select section</label>
                        <div class="input-group">
                            <span class="input-group-text"><i class="bi bi-grid-3x3-gap"></i></span>
                            <select id="section" name="section" class="form-select" required>
                                <option value="mediamap" selected>MediaMap</option>
                                <option value="healthpin">HealthPIN</option>
                                <option value="admin">Admin</option>
                            </select>
                        </div>
                    </div>''', '')
    
    # Update the title and description
    updated_content = updated_content.replace('<span>AIMAP</span>', '<span>MediaMap Platform</span>')
    updated_content = updated_content.replace('<p>Sign in to your AI workspace</p>', '<p>Access your MediaMap and HealthPIN applications</p>')
    
    # Write back the updated template
    with open('backend/templates/login.html', 'w') as f:
        f.write(updated_content)
    
    print("✅ Updated login template")

def create_app_routes_file():
    """Create the routes file for app context management"""
    
    app_routes = '''"""
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
        """Set the app context and redirect to appropriate dashboard"""
        
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
    
    print("✅ Registered app context management routes")
'''
    
    with open('backend/app_routes.py', 'w') as f:
        f.write(app_routes)
    
    print("✅ Created app routes file")

def create_user_dashboard_templates():
    """Create user dashboard templates for MediaMap and HealthPIN"""
    
    # MediaMap User Dashboard
    mediamap_user_dashboard = '''{% extends "base.html" %}

{% block title %}MediaMap Dashboard{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <div>
                    <h1 class="h3 mb-0">
                        <i class="bi bi-newspaper text-primary me-2"></i>
                        MediaMap Dashboard
                    </h1>
                    <p class="text-muted mb-0">Media analysis and content creation tools</p>
                </div>
                <div>
                    <a href="{{ url_for('app_selector') }}" class="btn btn-outline-secondary">
                        <i class="bi bi-grid-3x3-gap me-2"></i>Switch App
                    </a>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Quick Stats -->
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card bg-primary text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_analyses or 0 }}</h4>
                            <p class="mb-0">Media Analyses</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-graph-up" style="font-size: 2rem;"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-3">
            <div class="card bg-success text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_content or 0 }}</h4>
                            <p class="mb-0">Content Pieces</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-file-text" style="font-size: 2rem;"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-3">
            <div class="card bg-info text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_insights or 0 }}</h4>
                            <p class="mb-0">AI Insights</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-lightbulb" style="font-size: 2rem;"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-3">
            <div class="card bg-warning text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_reports or 0 }}</h4>
                            <p class="mb-0">Reports</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-file-earmark-text" style="font-size: 2rem;"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Main Actions -->
    <div class="row mb-4">
        <div class="col-md-6">
            <div class="card h-100">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0"><i class="bi bi-plus-circle me-2"></i>Create New</h5>
                </div>
                <div class="card-body">
                    <div class="row g-3">
                        <div class="col-6">
                            <a href="{{ url_for('new_media_analysis') }}" class="btn btn-outline-primary w-100">
                                <i class="bi bi-graph-up d-block mb-2" style="font-size: 2rem;"></i>
                                Media Analysis
                            </a>
                        </div>
                        <div class="col-6">
                            <a href="{{ url_for('new_content') }}" class="btn btn-outline-success w-100">
                                <i class="bi bi-file-text d-block mb-2" style="font-size: 2rem;"></i>
                                Content
                            </a>
                        </div>
                        <div class="col-6">
                            <a href="{{ url_for('new_report') }}" class="btn btn-outline-info w-100">
                                <i class="bi bi-file-earmark-text d-block mb-2" style="font-size: 2rem;"></i>
                                Report
                            </a>
                        </div>
                        <div class="col-6">
                            <a href="{{ url_for('ai_assistant') }}" class="btn btn-outline-warning w-100">
                                <i class="bi bi-robot d-block mb-2" style="font-size: 2rem;"></i>
                                AI Assistant
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card h-100">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0"><i class="bi bi-clock-history me-2"></i>Recent Activity</h5>
                </div>
                <div class="card-body">
                    {% if recent_activities %}
                        {% for activity in recent_activities %}
                        <div class="d-flex align-items-center mb-3">
                            <div class="flex-shrink-0">
                                <i class="bi bi-{{ activity.icon }} text-{{ activity.color }}"></i>
                            </div>
                            <div class="flex-grow-1 ms-3">
                                <h6 class="mb-0">{{ activity.title }}</h6>
                                <small class="text-muted">{{ activity.time }}</small>
                            </div>
                        </div>
                        {% endfor %}
                    {% else %}
                        <div class="text-center py-4">
                            <i class="bi bi-inbox display-4 text-muted"></i>
                            <p class="text-muted mt-2">No recent activity</p>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # HealthPIN User Dashboard
    healthpin_user_dashboard = '''{% extends "base.html" %}

{% block title %}HealthPIN Dashboard{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <div>
                    <h1 class="h3 mb-0">
                        <i class="bi bi-heart-pulse-fill text-success me-2"></i>
                        HealthPIN Dashboard
                    </h1>
                    <p class="text-muted mb-0">Healthcare data analysis and patient matching</p>
                </div>
                <div>
                    <a href="{{ url_for('app_selector') }}" class="btn btn-outline-secondary">
                        <i class="bi bi-grid-3x3-gap me-2"></i>Switch App
                    </a>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Quick Stats -->
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card bg-success text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_patients or 0 }}</h4>
                            <p class="mb-0">Patients</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-person-heart" style="font-size: 2rem;"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-3">
            <div class="card bg-primary text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_doctors or 0 }}</h4>
                            <p class="mb-0">Doctors</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-person-badge" style="font-size: 2rem;"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-3">
            <div class="card bg-info text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_records or 0 }}</h4>
                            <p class="mb-0">Medical Records</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-file-medical" style="font-size: 2rem;"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-3">
            <div class="card bg-warning text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_matches or 0 }}</h4>
                            <p class="mb-0">Matches</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-heart-arrow" style="font-size: 2rem;"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Main Actions -->
    <div class="row mb-4">
        <div class="col-md-6">
            <div class="card h-100">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0"><i class="bi bi-plus-circle me-2"></i>Quick Actions</h5>
                </div>
                <div class="card-body">
                    <div class="row g-3">
                        <div class="col-6">
                            <a href="{{ url_for('find_doctor') }}" class="btn btn-outline-primary w-100">
                                <i class="bi bi-search d-block mb-2" style="font-size: 2rem;"></i>
                                Find Doctor
                            </a>
                        </div>
                        <div class="col-6">
                            <a href="{{ url_for('patient_matching') }}" class="btn btn-outline-success w-100">
                                <i class="bi bi-heart-arrow d-block mb-2" style="font-size: 2rem;"></i>
                                Patient Matching
                            </a>
                        </div>
                        <div class="col-6">
                            <a href="{{ url_for('health_insights') }}" class="btn btn-outline-info w-100">
                                <i class="bi bi-lightbulb d-block mb-2" style="font-size: 2rem;"></i>
                                Health Insights
                            </a>
                        </div>
                        <div class="col-6">
                            <a href="{{ url_for('medical_records') }}" class="btn btn-outline-warning w-100">
                                <i class="bi bi-file-medical d-block mb-2" style="font-size: 2rem;"></i>
                                Medical Records
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card h-100">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0"><i class="bi bi-activity me-2"></i>Health Activity</h5>
                </div>
                <div class="card-body">
                    {% if health_activities %}
                        {% for activity in health_activities %}
                        <div class="d-flex align-items-center mb-3">
                            <div class="flex-shrink-0">
                                <i class="bi bi-{{ activity.icon }} text-{{ activity.color }}"></i>
                            </div>
                            <div class="flex-grow-1 ms-3">
                                <h6 class="mb-0">{{ activity.title }}</h6>
                                <small class="text-muted">{{ activity.time }}</small>
                            </div>
                        </div>
                        {% endfor %}
                    {% else %}
                        <div class="text-center py-4">
                            <i class="bi bi-heart display-4 text-muted"></i>
                            <p class="text-muted mt-2">No recent health activity</p>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Create directories
    os.makedirs('backend/templates/mediamap', exist_ok=True)
    os.makedirs('backend/templates/healthpin', exist_ok=True)
    
    # Write templates
    with open('backend/templates/mediamap/user_dashboard.html', 'w') as f:
        f.write(mediamap_user_dashboard)
    
    with open('backend/templates/healthpin/user_dashboard.html', 'w') as f:
        f.write(healthpin_user_dashboard)
    
    print("✅ Created user dashboard templates")

def main():
    """Main function to create the multi-app architecture"""
    
    print("🚀 CREATING MULTI-APP ARCHITECTURE")
    print("==================================")
    
    # Create all components
    create_app_selector_template()
    create_filtered_sidebar_templates()
    update_login_template()
    create_app_routes_file()
    create_user_dashboard_templates()
    
    print("")
    print("✅ MULTI-APP ARCHITECTURE CREATED!")
    print("==================================")
    print("")
    print("🎯 What's been created:")
    print("• App selector page with 4 app options")
    print("• Filtered sidebars for MediaMap Admin and HealthPIN Admin")
    print("• Updated login template (removed section selector)")
    print("• App context management routes")
    print("• User dashboard templates for MediaMap and HealthPIN")
    print("")
    print("🔧 Next steps:")
    print("1. Import and register app_routes in app.py")
    print("2. Update admin templates to use filtered sidebars")
    print("3. Update login route to redirect to app selector")
    print("4. Test the multi-app flow")

if __name__ == "__main__":
    main()
