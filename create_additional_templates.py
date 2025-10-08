#!/usr/bin/env python3
"""
Create Additional Templates for Separate Admin Apps
===================================================

Creates all the necessary templates for MediaMap Admin and HealthPIN Admin apps.
"""

import os

def create_base_admin_template():
    """Create a base admin template for the separate admin apps"""
    
    base_admin_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}DEVELOP AI Admin{% endblock %}</title>
    
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
        
        .badge {
            border-radius: 10px;
        }
        
        .table {
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
    
    {% block extra_css %}{% endblock %}
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                {% if 'mediamap' in request.endpoint %}
                    <i class="bi bi-newspaper me-2"></i>DEVELOP AI - MediaMap Admin
                {% elif 'healthpin' in request.endpoint %}
                    <i class="bi bi-heart-pulse-fill me-2"></i>DEVELOP AI - HealthPIN Admin
                {% else %}
                    <i class="bi bi-cpu-fill me-2"></i>DEVELOP AI Admin
                {% endif %}
            </a>
            
            <div class="navbar-nav ms-auto">
                <div class="nav-item dropdown">
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
                </div>
            </div>
        </div>
    </nav>
    
    <!-- Main Content -->
    <main class="main-content">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="container-fluid">
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
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    
    {% block extra_js %}{% endblock %}
</body>
</html>'''
    
    with open('backend/templates/base_admin.html', 'w') as f:
        f.write(base_admin_template)
    
    print("✅ Created base admin template")

def create_mediamap_admin_templates():
    """Create all MediaMap Admin templates"""
    
    print("📰 Creating MediaMap Admin templates...")
    
    # Media Analysis Template
    media_analysis_template = '''{% extends "base_admin.html" %}

{% block title %}Media Analysis - MediaMap Admin{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row mb-4">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1 class="h3 mb-0">
                        <i class="bi bi-graph-up text-primary me-2"></i>
                        Media Analysis
                    </h1>
                    <p class="text-muted mb-0">Analyze media content and sentiment</p>
                </div>
                <div>
                    <button class="btn btn-primary">
                        <i class="bi bi-plus-circle me-2"></i>New Analysis
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-12">
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">Recent Analyses</h5>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Title</th>
                                    <th>Status</th>
                                    <th>Date</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for analysis in analyses %}
                                <tr>
                                    <td>{{ analysis.title }}</td>
                                    <td>
                                        <span class="badge bg-{% if analysis.status == 'completed' %}success{% elif analysis.status == 'in_progress' %}warning{% else %}secondary{% endif %}">
                                            {{ analysis.status.replace('_', ' ').title() }}
                                        </span>
                                    </td>
                                    <td>{{ analysis.date }}</td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-primary">View</button>
                                        <button class="btn btn-sm btn-outline-secondary">Edit</button>
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open('backend/templates/admin_apps/mediamap_admin/media_analysis.html', 'w') as f:
        f.write(media_analysis_template)
    
    # Content Management Template
    content_management_template = '''{% extends "base_admin.html" %}

{% block title %}Content Management - MediaMap Admin{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row mb-4">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1 class="h3 mb-0">
                        <i class="bi bi-file-text text-success me-2"></i>
                        Content Management
                    </h1>
                    <p class="text-muted mb-0">Manage articles, reports, and media content</p>
                </div>
                <div>
                    <button class="btn btn-success">
                        <i class="bi bi-plus-circle me-2"></i>New Content
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-12">
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">Content Items</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        {% for item in content_items %}
                        <div class="col-md-4 mb-3">
                            <div class="card h-100">
                                <div class="card-body">
                                    <h6 class="card-title">{{ item.title }}</h6>
                                    <p class="card-text">
                                        <span class="badge bg-info">{{ item.type.title() }}</span>
                                        <span class="badge bg-{% if item.status == 'published' %}success{% elif item.status == 'draft' %}warning{% else %}secondary{% endif %} ms-2">
                                            {{ item.status.title() }}
                                        </span>
                                    </p>
                                    <small class="text-muted">{{ item.date }}</small>
                                </div>
                                <div class="card-footer">
                                    <button class="btn btn-sm btn-outline-primary">Edit</button>
                                    <button class="btn btn-sm btn-outline-success">Publish</button>
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open('backend/templates/admin_apps/mediamap_admin/content_management.html', 'w') as f:
        f.write(content_management_template)
    
    print("✅ MediaMap Admin templates created")

def create_healthpin_admin_templates():
    """Create all HealthPIN Admin templates"""
    
    print("🏥 Creating HealthPIN Admin templates...")
    
    # Patients Template
    patients_template = '''{% extends "base_admin.html" %}

{% block title %}Patient Management - HealthPIN Admin{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row mb-4">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1 class="h3 mb-0">
                        <i class="bi bi-person-heart text-success me-2"></i>
                        Patient Management
                    </h1>
                    <p class="text-muted mb-0">Manage patient data and clinical cases ({{ total_count }} total)</p>
                </div>
                <div>
                    <button class="btn btn-success">
                        <i class="bi bi-plus-circle me-2"></i>Add Patient
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-12">
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">Patient Cases</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        {% for patient in patients %}
                        <div class="col-md-6 col-lg-4 mb-3">
                            <div class="card h-100 border-start border-4 border-success">
                                <div class="card-body">
                                    <h6 class="card-title">{{ patient.name }}</h6>
                                    <p class="card-text small">{{ patient.condition }}</p>
                                    <div class="d-flex justify-content-between align-items-center">
                                        <small class="text-muted">{{ patient.date }}</small>
                                        <span class="badge bg-{% if patient.status == 'active' %}success{% else %}secondary{% endif %}">
                                            {{ patient.status.title() }}
                                        </span>
                                    </div>
                                    <div class="mt-2">
                                        <small class="text-muted">
                                            <i class="bi bi-building me-1"></i>{{ patient.source }}
                                        </small>
                                    </div>
                                </div>
                                <div class="card-footer">
                                    <button class="btn btn-sm btn-outline-primary">View</button>
                                    <button class="btn btn-sm btn-outline-success">Match</button>
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open('backend/templates/admin_apps/healthpin_admin/patients.html', 'w') as f:
        f.write(patients_template)
    
    # Doctors Template
    doctors_template = '''{% extends "base_admin.html" %}

{% block title %}Doctor Management - HealthPIN Admin{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row mb-4">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1 class="h3 mb-0">
                        <i class="bi bi-person-badge text-primary me-2"></i>
                        Doctor Management
                    </h1>
                    <p class="text-muted mb-0">Manage healthcare professionals ({{ total_count }} total)</p>
                </div>
                <div>
                    <button class="btn btn-primary">
                        <i class="bi bi-plus-circle me-2"></i>Add Doctor
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-12">
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">Healthcare Professionals</h5>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Specialty</th>
                                    <th>Location</th>
                                    <th>Patients</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for doctor in doctors %}
                                <tr>
                                    <td>
                                        <div class="d-flex align-items-center">
                                            <i class="bi bi-person-circle me-2 text-primary"></i>
                                            {{ doctor.name }}
                                        </div>
                                    </td>
                                    <td>{{ doctor.specialty }}</td>
                                    <td>{{ doctor.location }}</td>
                                    <td>{{ doctor.patients }}</td>
                                    <td>
                                        <span class="badge bg-{% if doctor.verified %}success{% else %}warning{% endif %}">
                                            {% if doctor.verified %}Verified{% else %}Pending{% endif %}
                                        </span>
                                    </td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-primary">View</button>
                                        <button class="btn btn-sm btn-outline-success">Contact</button>
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open('backend/templates/admin_apps/healthpin_admin/doctors.html', 'w') as f:
        f.write(doctors_template)
    
    print("✅ HealthPIN Admin templates created")

def create_init_files():
    """Create __init__.py files for the admin apps"""
    
    print("📝 Creating __init__.py files...")
    
    # MediaMap Admin __init__.py
    with open('backend/admin_apps/__init__.py', 'w') as f:
        f.write('# Admin Apps Package\n')
    
    with open('backend/admin_apps/mediamap_admin/__init__.py', 'w') as f:
        f.write('# MediaMap Admin App\n')
    
    with open('backend/admin_apps/healthpin_admin/__init__.py', 'w') as f:
        f.write('# HealthPIN Admin App\n')
    
    print("✅ __init__.py files created")

def main():
    """Main function to create additional templates"""
    
    print("🎨 CREATING ADDITIONAL TEMPLATES")
    print("===============================")
    
    create_base_admin_template()
    create_mediamap_admin_templates()
    create_healthpin_admin_templates()
    create_init_files()
    
    print("")
    print("✅ ADDITIONAL TEMPLATES CREATED!")
    print("===============================")
    print("")
    print("🎯 Templates created:")
    print("• base_admin.html - Base template for admin apps")
    print("• MediaMap Admin templates (media analysis, content management)")
    print("• HealthPIN Admin templates (patients, doctors)")
    print("• __init__.py files for proper Python packages")
    print("")
    print("🔄 Ready to restart and test the separate admin apps!")

if __name__ == "__main__":
    main()
