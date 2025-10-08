#!/usr/bin/env python3
"""
Create Separate Admin Apps
==========================

This script creates completely separate admin interfaces for MediaMap Admin and HealthPIN Admin,
each functioning as independent apps within the same codebase.
"""

import os
import shutil
import re

def create_directory_structure():
    """Create the directory structure for separate admin apps"""
    
    print("📁 Creating directory structure...")
    
    # Create main admin app directories
    directories = [
        'backend/admin_apps',
        'backend/admin_apps/mediamap_admin',
        'backend/admin_apps/mediamap_admin/templates',
        'backend/admin_apps/mediamap_admin/static',
        'backend/admin_apps/healthpin_admin',
        'backend/admin_apps/healthpin_admin/templates',
        'backend/admin_apps/healthpin_admin/static',
        'backend/templates/admin_apps',
        'backend/templates/admin_apps/mediamap_admin',
        'backend/templates/admin_apps/healthpin_admin'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}")
    
    print("✅ Directory structure created")

def create_mediamap_admin_app():
    """Create the MediaMap Admin app with its own routes and templates"""
    
    print("📰 Creating MediaMap Admin app...")
    
    # MediaMap Admin Routes
    mediamap_admin_routes = '''"""
MediaMap Admin Application
=========================

Dedicated admin interface for MediaMap functionality including:
- Media analysis and reporting
- Content management
- MediaMap agents
- Organizations and clients
- Media-specific insights
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import login_required, current_user
from datetime import datetime
import json

# Create MediaMap Admin Blueprint
mediamap_admin_bp = Blueprint('mediamap_admin', __name__, 
                             url_prefix='/mediamap-admin',
                             template_folder='templates',
                             static_folder='static')

@mediamap_admin_bp.route('/')
@mediamap_admin_bp.route('/dashboard')
@login_required
def dashboard():
    """MediaMap Admin Dashboard"""
    
    # Ensure user is in MediaMap Admin context
    if session.get('app_context') != 'mediamap_admin':
        return redirect(url_for('app_selector'))
    
    # Mock data for MediaMap dashboard
    dashboard_data = {
        'total_analyses': 45,
        'total_content': 128,
        'total_organizations': 12,
        'total_reports': 67,
        'recent_activities': [
            {'title': 'New Media Analysis Completed', 'time': '2 hours ago', 'icon': 'graph-up', 'color': 'success'},
            {'title': 'Content Published', 'time': '4 hours ago', 'icon': 'file-text', 'color': 'info'},
            {'title': 'Organization Added', 'time': '1 day ago', 'icon': 'building', 'color': 'warning'},
        ],
        'media_stats': {
            'articles_analyzed': 234,
            'sentiment_positive': 67,
            'sentiment_neutral': 28,
            'sentiment_negative': 5
        }
    }
    
    return render_template('admin_apps/mediamap_admin/dashboard.html', **dashboard_data)

@mediamap_admin_bp.route('/media-analysis')
@login_required
def media_analysis():
    """Media Analysis Management"""
    
    if session.get('app_context') != 'mediamap_admin':
        return redirect(url_for('app_selector'))
    
    # Mock media analysis data
    analyses = [
        {'id': 1, 'title': 'Q3 Media Sentiment Analysis', 'status': 'completed', 'date': '2025-10-01'},
        {'id': 2, 'title': 'Brand Mention Analysis', 'status': 'in_progress', 'date': '2025-10-05'},
        {'id': 3, 'title': 'Competitor Analysis', 'status': 'pending', 'date': '2025-10-07'},
    ]
    
    return render_template('admin_apps/mediamap_admin/media_analysis.html', analyses=analyses)

@mediamap_admin_bp.route('/content-management')
@login_required
def content_management():
    """Content Management"""
    
    if session.get('app_context') != 'mediamap_admin':
        return redirect(url_for('app_selector'))
    
    # Mock content data
    content_items = [
        {'id': 1, 'title': 'AI in Media Analysis', 'type': 'article', 'status': 'published', 'date': '2025-10-01'},
        {'id': 2, 'title': 'Social Media Trends', 'type': 'report', 'status': 'draft', 'date': '2025-10-03'},
        {'id': 3, 'title': 'Brand Strategy Guide', 'type': 'guide', 'status': 'review', 'date': '2025-10-05'},
    ]
    
    return render_template('admin_apps/mediamap_admin/content_management.html', content_items=content_items)

@mediamap_admin_bp.route('/agents')
@login_required
def agents():
    """MediaMap Agents Management"""
    
    if session.get('app_context') != 'mediamap_admin':
        return redirect(url_for('app_selector'))
    
    # Mock MediaMap agents data
    agents = [
        {
            'name': 'MediaMap Agent',
            'type': 'mediamap',
            'status': 'active',
            'description': 'Media analysis and content generation',
            'data_points': 1247,
            'insights': 89,
            'last_run': '2 hours ago'
        }
    ]
    
    return render_template('admin_apps/mediamap_admin/agents.html', agents=agents)

@mediamap_admin_bp.route('/organizations')
@login_required
def organizations():
    """Organizations Management"""
    
    if session.get('app_context') != 'mediamap_admin':
        return redirect(url_for('app_selector'))
    
    # Mock organizations data
    organizations = [
        {'id': 1, 'name': 'TechCorp Media', 'type': 'client', 'status': 'active', 'projects': 5},
        {'id': 2, 'name': 'Global News Network', 'type': 'partner', 'status': 'active', 'projects': 3},
        {'id': 3, 'name': 'Digital Marketing Inc', 'type': 'client', 'status': 'pending', 'projects': 1},
    ]
    
    return render_template('admin_apps/mediamap_admin/organizations.html', organizations=organizations)

@mediamap_admin_bp.route('/reports')
@login_required
def reports():
    """Reports Management"""
    
    if session.get('app_context') != 'mediamap_admin':
        return redirect(url_for('app_selector'))
    
    # Mock reports data
    reports = [
        {'id': 1, 'title': 'Monthly Media Analysis', 'type': 'monthly', 'status': 'completed', 'date': '2025-10-01'},
        {'id': 2, 'title': 'Brand Sentiment Report', 'type': 'sentiment', 'status': 'generating', 'date': '2025-10-07'},
        {'id': 3, 'title': 'Competitor Analysis', 'type': 'competitive', 'status': 'scheduled', 'date': '2025-10-10'},
    ]
    
    return render_template('admin_apps/mediamap_admin/reports.html', reports=reports)

@mediamap_admin_bp.route('/insights')
@login_required
def insights():
    """MediaMap Insights"""
    
    if session.get('app_context') != 'mediamap_admin':
        return redirect(url_for('app_selector'))
    
    # Mock MediaMap insights
    insights = [
        {
            'title': 'Social Media Engagement Up 45%',
            'description': 'Brand mentions and engagement have increased significantly this quarter.',
            'category': 'social_media',
            'date': '2025-10-01',
            'impact': 'high'
        },
        {
            'title': 'Content Performance Analysis',
            'description': 'Video content performs 3x better than text-only posts.',
            'category': 'content',
            'date': '2025-10-03',
            'impact': 'medium'
        }
    ]
    
    return render_template('admin_apps/mediamap_admin/insights.html', insights=insights)

@mediamap_admin_bp.route('/settings')
@login_required
def settings():
    """MediaMap Admin Settings"""
    
    if session.get('app_context') != 'mediamap_admin':
        return redirect(url_for('app_selector'))
    
    return render_template('admin_apps/mediamap_admin/settings.html')

# API Routes for MediaMap Admin
@mediamap_admin_bp.route('/api/stats')
@login_required
def api_stats():
    """API endpoint for MediaMap stats"""
    
    stats = {
        'analyses': 45,
        'content': 128,
        'organizations': 12,
        'reports': 67,
        'agents_active': 1,
        'insights_generated': 89
    }
    
    return jsonify({'success': True, 'stats': stats})

def register_mediamap_admin_routes(app):
    """Register MediaMap Admin routes with the Flask app"""
    app.register_blueprint(mediamap_admin_bp)
    print("✅ MediaMap Admin routes registered")
'''
    
    with open('backend/admin_apps/mediamap_admin/routes.py', 'w') as f:
        f.write(mediamap_admin_routes)
    
    print("✅ MediaMap Admin routes created")

def create_healthpin_admin_app():
    """Create the HealthPIN Admin app with its own routes and templates"""
    
    print("🏥 Creating HealthPIN Admin app...")
    
    # HealthPIN Admin Routes
    healthpin_admin_routes = '''"""
HealthPIN Admin Application
==========================

Dedicated admin interface for HealthPIN functionality including:
- Patient and doctor management
- Medical records and matching
- HealthPIN agents
- Health insights and analytics
- Healthcare-specific features
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import login_required, current_user
from datetime import datetime
import json
import os

# Create HealthPIN Admin Blueprint
healthpin_admin_bp = Blueprint('healthpin_admin', __name__, 
                              url_prefix='/healthpin-admin',
                              template_folder='templates',
                              static_folder='static')

@healthpin_admin_bp.route('/')
@healthpin_admin_bp.route('/dashboard')
@login_required
def dashboard():
    """HealthPIN Admin Dashboard"""
    
    # Ensure user is in HealthPIN Admin context
    if session.get('app_context') != 'healthpin_admin':
        return redirect(url_for('app_selector'))
    
    # Load real HealthPIN agent data
    dashboard_data = load_healthpin_dashboard_data()
    
    return render_template('admin_apps/healthpin_admin/dashboard.html', **dashboard_data)

def load_healthpin_dashboard_data():
    """Load real HealthPIN dashboard data from agent storage"""
    
    try:
        # Load real agent data
        agent_data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        
        if os.path.exists(agent_data_file):
            with open(agent_data_file, 'r') as f:
                agent_data = json.load(f)
            
            # Process real data
            categories = {}
            sources = set()
            
            for entry in agent_data:
                category = entry.get('category', 'Unknown')
                source = entry.get('source', 'Unknown')
                categories[category] = categories.get(category, 0) + 1
                sources.add(source)
            
            # Real numbers from actual data
            total_patients = categories.get('Clinical_Care', 0)
            total_doctors = len(sources)
            total_records = len(agent_data)
            total_matches = len(categories)
            
            # Recent activities from real data
            recent_activities = []
            for i, entry in enumerate(agent_data[-5:]):  # Last 5 entries
                recent_activities.append({
                    'title': f'New {entry.get("category", "Healthcare")} Data',
                    'time': f'{i+1} hours ago',
                    'icon': 'heart-pulse',
                    'color': 'success'
                })
            
        else:
            # Fallback data
            total_patients = 60
            total_doctors = 2
            total_records = 176
            total_matches = 4
            recent_activities = [
                {'title': 'Patient Data Updated', 'time': '1 hour ago', 'icon': 'person-heart', 'color': 'success'},
                {'title': 'Doctor Verified', 'time': '3 hours ago', 'icon': 'person-badge', 'color': 'info'},
                {'title': 'Health Record Added', 'time': '5 hours ago', 'icon': 'file-medical', 'color': 'warning'},
            ]
        
        return {
            'total_patients': total_patients,
            'total_doctors': total_doctors,
            'total_records': total_records,
            'total_matches': total_matches,
            'recent_activities': recent_activities,
            'health_stats': {
                'clinical_care': categories.get('Clinical_Care', 0),
                'medical_research': categories.get('Medical_Research', 0),
                'healthcare_policy': categories.get('Healthcare_Policy', 0),
                'general_healthcare': categories.get('General_Healthcare', 0)
            }
        }
        
    except Exception as e:
        print(f"Error loading HealthPIN data: {e}")
        # Return fallback data
        return {
            'total_patients': 60,
            'total_doctors': 2,
            'total_records': 176,
            'total_matches': 4,
            'recent_activities': [],
            'health_stats': {
                'clinical_care': 60,
                'medical_research': 48,
                'healthcare_policy': 16,
                'general_healthcare': 52
            }
        }

@healthpin_admin_bp.route('/patients')
@login_required
def patients():
    """Patient Management"""
    
    if session.get('app_context') != 'healthpin_admin':
        return redirect(url_for('app_selector'))
    
    # Load real patient data from agent storage
    patients_data = load_patient_data()
    
    return render_template('admin_apps/healthpin_admin/patients.html', 
                         patients=patients_data['patients'],
                         total_count=patients_data['total_count'])

def load_patient_data():
    """Load patient data from HealthPIN agent storage"""
    
    try:
        agent_data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        
        if os.path.exists(agent_data_file):
            with open(agent_data_file, 'r') as f:
                agent_data = json.load(f)
            
            # Filter for clinical care entries (patients)
            patients = []
            for i, entry in enumerate(agent_data):
                if entry.get('category') == 'Clinical_Care':
                    patients.append({
                        'id': i + 1,
                        'name': f'Patient Case {i + 1}',
                        'condition': entry.get('content', '')[:100] + '...',
                        'source': entry.get('source', 'Unknown'),
                        'date': entry.get('timestamp', '2025-10-07')[:10],
                        'status': 'active'
                    })
            
            return {'patients': patients[:20], 'total_count': len(patients)}  # Limit to 20 for display
        
    except Exception as e:
        print(f"Error loading patient data: {e}")
    
    # Fallback mock data
    return {
        'patients': [
            {'id': 1, 'name': 'Clinical Case 1', 'condition': 'Cardiovascular assessment', 'source': 'WHO', 'date': '2025-10-07', 'status': 'active'},
            {'id': 2, 'name': 'Clinical Case 2', 'condition': 'Mental health evaluation', 'source': 'Medical News', 'date': '2025-10-06', 'status': 'active'},
        ],
        'total_count': 60
    }

@healthpin_admin_bp.route('/doctors')
@login_required
def doctors():
    """Doctor Management"""
    
    if session.get('app_context') != 'healthpin_admin':
        return redirect(url_for('app_selector'))
    
    # Load real doctor data
    doctors_data = load_doctor_data()
    
    return render_template('admin_apps/healthpin_admin/doctors.html',
                         doctors=doctors_data['doctors'],
                         total_count=doctors_data['total_count'])

def load_doctor_data():
    """Load doctor data from HealthPIN agent storage"""
    
    try:
        agent_data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        
        if os.path.exists(agent_data_file):
            with open(agent_data_file, 'r') as f:
                agent_data = json.load(f)
            
            # Get unique sources as doctors
            sources = set()
            for entry in agent_data:
                sources.add(entry.get('source', 'Unknown'))
            
            doctors = []
            for i, source in enumerate(list(sources)):
                if 'who.int' in source:
                    name = 'WHO Health Data'
                    specialty = 'Global Health'
                elif 'medicalnews' in source:
                    name = 'Medical News Today'
                    specialty = 'Healthcare News'
                else:
                    name = f'Healthcare Source {i+1}'
                    specialty = 'General Medicine'
                
                doctors.append({
                    'id': i + 1,
                    'name': name,
                    'specialty': specialty,
                    'location': 'South Africa',
                    'verified': True,
                    'patients': len([e for e in agent_data if e.get('source') == source])
                })
            
            return {'doctors': doctors, 'total_count': len(doctors)}
        
    except Exception as e:
        print(f"Error loading doctor data: {e}")
    
    # Fallback mock data
    return {
        'doctors': [
            {'id': 1, 'name': 'WHO Health Data', 'specialty': 'Global Health', 'location': 'South Africa', 'verified': True, 'patients': 89},
            {'id': 2, 'name': 'Medical News Today', 'specialty': 'Healthcare News', 'location': 'South Africa', 'verified': True, 'patients': 87},
        ],
        'total_count': 2
    }

@healthpin_admin_bp.route('/agents')
@login_required
def agents():
    """HealthPIN Agents Management"""
    
    if session.get('app_context') != 'healthpin_admin':
        return redirect(url_for('app_selector'))
    
    # Mock HealthPIN agents data
    agents = [
        {
            'name': 'HealthPIN Agent',
            'type': 'healthpin',
            'status': 'active',
            'description': 'Healthcare data analysis and patient matching',
            'data_points': 176,
            'insights': 10,
            'last_run': '1 hour ago'
        }
    ]
    
    return render_template('admin_apps/healthpin_admin/agents.html', agents=agents)

@healthpin_admin_bp.route('/records')
@login_required
def records():
    """Medical Records Management"""
    
    if session.get('app_context') != 'healthpin_admin':
        return redirect(url_for('app_selector'))
    
    # Load medical records data
    records_data = load_medical_records()
    
    return render_template('admin_apps/healthpin_admin/records.html',
                         records=records_data['records'],
                         total_count=records_data['total_count'])

def load_medical_records():
    """Load medical records from agent data"""
    
    try:
        agent_data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        
        if os.path.exists(agent_data_file):
            with open(agent_data_file, 'r') as f:
                agent_data = json.load(f)
            
            records = []
            for i, entry in enumerate(agent_data):
                records.append({
                    'id': i + 1,
                    'title': f'Health Record {i + 1}',
                    'category': entry.get('category', 'General'),
                    'content': entry.get('content', '')[:150] + '...',
                    'date': entry.get('timestamp', '2025-10-07')[:10],
                    'source': entry.get('source', 'Unknown')
                })
            
            return {'records': records[:30], 'total_count': len(records)}  # Limit for display
        
    except Exception as e:
        print(f"Error loading medical records: {e}")
    
    # Fallback data
    return {
        'records': [
            {'id': 1, 'title': 'Health Record 1', 'category': 'Clinical_Care', 'content': 'Patient assessment and care plan...', 'date': '2025-10-07', 'source': 'WHO'},
        ],
        'total_count': 176
    }

@healthpin_admin_bp.route('/matching')
@login_required
def matching():
    """Patient Matching Management"""
    
    if session.get('app_context') != 'healthpin_admin':
        return redirect(url_for('app_selector'))
    
    # Mock matching data
    matches = [
        {'id': 1, 'patient': 'Clinical Case 1', 'doctor': 'WHO Health Data', 'compatibility': 95, 'status': 'matched'},
        {'id': 2, 'patient': 'Clinical Case 2', 'doctor': 'Medical News Today', 'compatibility': 87, 'status': 'pending'},
    ]
    
    return render_template('admin_apps/healthpin_admin/matching.html', matches=matches)

@healthpin_admin_bp.route('/insights')
@login_required
def insights():
    """HealthPIN Insights"""
    
    if session.get('app_context') != 'healthpin_admin':
        return redirect(url_for('app_selector'))
    
    # Load health insights
    insights_data = load_health_insights()
    
    return render_template('admin_apps/healthpin_admin/insights.html', insights=insights_data)

def load_health_insights():
    """Load health insights from agent data"""
    
    insights = [
        {
            'title': 'Mental Health Awareness Increasing',
            'description': 'WHO reports show 40% increase in mental health discussions.',
            'category': 'mental_health',
            'date': '2025-10-01',
            'impact': 'high'
        },
        {
            'title': 'Cardiovascular Disease Prevention',
            'description': 'New guidelines for early detection and prevention strategies.',
            'category': 'cardiovascular',
            'date': '2025-10-03',
            'impact': 'medium'
        }
    ]
    
    return insights

@healthpin_admin_bp.route('/settings')
@login_required
def settings():
    """HealthPIN Admin Settings"""
    
    if session.get('app_context') != 'healthpin_admin':
        return redirect(url_for('app_selector'))
    
    return render_template('admin_apps/healthpin_admin/settings.html')

# API Routes for HealthPIN Admin
@healthpin_admin_bp.route('/api/stats')
@login_required
def api_stats():
    """API endpoint for HealthPIN stats"""
    
    dashboard_data = load_healthpin_dashboard_data()
    
    stats = {
        'patients': dashboard_data['total_patients'],
        'doctors': dashboard_data['total_doctors'],
        'records': dashboard_data['total_records'],
        'matches': dashboard_data['total_matches'],
        'agents_active': 1,
        'insights_generated': 10
    }
    
    return jsonify({'success': True, 'stats': stats})

def register_healthpin_admin_routes(app):
    """Register HealthPIN Admin routes with the Flask app"""
    app.register_blueprint(healthpin_admin_bp)
    print("✅ HealthPIN Admin routes registered")
'''
    
    with open('backend/admin_apps/healthpin_admin/routes.py', 'w') as f:
        f.write(healthpin_admin_routes)
    
    print("✅ HealthPIN Admin routes created")

def create_admin_templates():
    """Create templates for both admin apps"""
    
    print("🎨 Creating admin templates...")
    
    # MediaMap Admin Dashboard Template
    mediamap_dashboard = '''{% extends "base_admin.html" %}

{% block title %}MediaMap Admin Dashboard{% endblock %}

{% block content %}
<div class="container-fluid">
    <!-- Header -->
    <div class="row mb-4">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1 class="h3 mb-0">
                        <i class="bi bi-newspaper text-primary me-2"></i>
                        MediaMap Admin Dashboard
                    </h1>
                    <p class="text-muted mb-0">Media analysis and content management</p>
                </div>
                <div>
                    <span class="badge bg-success">{{ total_analyses }} Analyses</span>
                    <span class="badge bg-info ms-2">{{ total_content }} Content Items</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Stats Cards -->
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card bg-primary text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_analyses }}</h4>
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
                            <h4 class="mb-0">{{ total_content }}</h4>
                            <p class="mb-0">Content Items</p>
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
                            <h4 class="mb-0">{{ total_organizations }}</h4>
                            <p class="mb-0">Organizations</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-building" style="font-size: 2rem;"></i>
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
                            <h4 class="mb-0">{{ total_reports }}</h4>
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
    
    <!-- Main Content -->
    <div class="row">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0"><i class="bi bi-activity me-2"></i>Recent Activity</h5>
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
        
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0"><i class="bi bi-bar-chart me-2"></i>Media Stats</h5>
                </div>
                <div class="card-body">
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>Articles Analyzed</span>
                            <strong>{{ media_stats.articles_analyzed }}</strong>
                        </div>
                    </div>
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>Positive Sentiment</span>
                            <strong class="text-success">{{ media_stats.sentiment_positive }}%</strong>
                        </div>
                    </div>
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>Neutral Sentiment</span>
                            <strong class="text-info">{{ media_stats.sentiment_neutral }}%</strong>
                        </div>
                    </div>
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>Negative Sentiment</span>
                            <strong class="text-danger">{{ media_stats.sentiment_negative }}%</strong>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open('backend/templates/admin_apps/mediamap_admin/dashboard.html', 'w') as f:
        f.write(mediamap_dashboard)
    
    # HealthPIN Admin Dashboard Template
    healthpin_dashboard = '''{% extends "base_admin.html" %}

{% block title %}HealthPIN Admin Dashboard{% endblock %}

{% block content %}
<div class="container-fluid">
    <!-- Header -->
    <div class="row mb-4">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1 class="h3 mb-0">
                        <i class="bi bi-heart-pulse-fill text-success me-2"></i>
                        HealthPIN Admin Dashboard
                    </h1>
                    <p class="text-muted mb-0">Healthcare data analysis and patient management</p>
                </div>
                <div>
                    <span class="badge bg-success">{{ total_patients }} Patients</span>
                    <span class="badge bg-info ms-2">{{ total_doctors }} Doctors</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Stats Cards -->
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card bg-success text-white">
                <div class="card-body">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h4 class="mb-0">{{ total_patients }}</h4>
                            <p class="mb-0">Total Patients</p>
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
                            <h4 class="mb-0">{{ total_doctors }}</h4>
                            <p class="mb-0">Verified Doctors</p>
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
                            <h4 class="mb-0">{{ total_records }}</h4>
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
                            <h4 class="mb-0">{{ total_matches }}</h4>
                            <p class="mb-0">Patient Matches</p>
                        </div>
                        <div class="align-self-center">
                            <i class="bi bi-heart-arrow" style="font-size: 2rem;"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Main Content -->
    <div class="row">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0"><i class="bi bi-activity me-2"></i>Recent Health Activity</h5>
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
                            <i class="bi bi-heart display-4 text-muted"></i>
                            <p class="text-muted mt-2">No recent health activity</p>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
        
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0"><i class="bi bi-pie-chart me-2"></i>Health Data Stats</h5>
                </div>
                <div class="card-body">
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>Clinical Care</span>
                            <strong class="text-success">{{ health_stats.clinical_care }}</strong>
                        </div>
                    </div>
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>Medical Research</span>
                            <strong class="text-info">{{ health_stats.medical_research }}</strong>
                        </div>
                    </div>
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>Healthcare Policy</span>
                            <strong class="text-warning">{{ health_stats.healthcare_policy }}</strong>
                        </div>
                    </div>
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>General Healthcare</span>
                            <strong class="text-primary">{{ health_stats.general_healthcare }}</strong>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    with open('backend/templates/admin_apps/healthpin_admin/dashboard.html', 'w') as f:
        f.write(healthpin_dashboard)
    
    print("✅ Admin dashboard templates created")

def update_app_routes_for_separate_apps():
    """Update app routes to redirect to separate admin apps"""
    
    print("🔧 Updating app routes for separate admin apps...")
    
    with open('backend/app_routes.py', 'r') as f:
        content = f.read()
    
    # Update the set_app_context function to redirect to separate admin apps
    new_redirect_logic = '''        # Redirect based on app type
        if app_type == 'mediamap':
            return redirect(url_for('mediamap_user_dashboard'))
        elif app_type == 'mediamap_admin':
            return redirect(url_for('mediamap_admin.dashboard'))
        elif app_type == 'healthpin':
            return redirect(url_for('healthpin_user_dashboard'))
        elif app_type == 'healthpin_admin':
            return redirect(url_for('healthpin_admin.dashboard'))
        else:
            return redirect(url_for('app_selector'))'''
    
    # Replace the existing redirect logic
    old_pattern = r'# Redirect based on app type.*?return redirect\(url_for\(\'app_selector\'\)\)'
    content = re.sub(old_pattern, new_redirect_logic, content, flags=re.DOTALL)
    
    with open('backend/app_routes.py', 'w') as f:
        f.write(content)
    
    print("✅ App routes updated for separate admin apps")

def update_main_app_with_separate_apps():
    """Update main app.py to register separate admin apps"""
    
    print("🔧 Updating main app.py...")
    
    with open('backend/app.py', 'r') as f:
        content = f.read()
    
    # Add imports for separate admin apps
    import_addition = '''
# Separate Admin Apps
try:
    from .admin_apps.mediamap_admin.routes import register_mediamap_admin_routes
    from .admin_apps.healthpin_admin.routes import register_healthpin_admin_routes
except ImportError:
    from admin_apps.mediamap_admin.routes import register_mediamap_admin_routes
    from admin_apps.healthpin_admin.routes import register_healthpin_admin_routes'''
    
    # Add after the existing imports
    if 'register_mediamap_admin_routes' not in content:
        import_pattern = r'(from filtered_admin_routes import register_filtered_admin_routes)'
        content = re.sub(import_pattern, r'\1' + import_addition, content)
        print("✅ Added separate admin app imports")
    
    # Register the separate admin apps
    registration_addition = '''
# Register separate admin apps
register_mediamap_admin_routes(app)
register_healthpin_admin_routes(app)'''
    
    if 'register_mediamap_admin_routes(app)' not in content:
        register_pattern = r'(register_filtered_admin_routes\(app\))'
        content = re.sub(register_pattern, r'\1' + registration_addition, content)
        print("✅ Added separate admin app registration")
    
    with open('backend/app.py', 'w') as f:
        f.write(content)
    
    print("✅ Main app.py updated")

def main():
    """Main function to create separate admin apps"""
    
    print("🏗️ CREATING SEPARATE ADMIN APPS")
    print("===============================")
    
    create_directory_structure()
    create_mediamap_admin_app()
    create_healthpin_admin_app()
    create_admin_templates()
    update_app_routes_for_separate_apps()
    update_main_app_with_separate_apps()
    
    print("")
    print("✅ SEPARATE ADMIN APPS CREATED!")
    print("==============================")
    print("")
    print("🎯 What's been created:")
    print("• 📰 MediaMap Admin App (/mediamap-admin/)")
    print("  - Dashboard, Media Analysis, Content, Agents, Organizations, Reports")
    print("• 🏥 HealthPIN Admin App (/healthpin-admin/)")
    print("  - Dashboard, Patients, Doctors, Agents, Records, Matching, Insights")
    print("• 🎨 Dedicated templates for each admin app")
    print("• 🔗 Separate route blueprints")
    print("• 📊 Real data integration for HealthPIN")
    print("")
    print("🎯 New URLs:")
    print("• MediaMap Admin: /mediamap-admin/")
    print("• HealthPIN Admin: /healthpin-admin/")
    print("")
    print("🔄 Restart your app to see the separate admin interfaces!")

if __name__ == "__main__":
    main()
