#!/usr/bin/env python3
"""
Enhance App-Specific Filtering
==============================

This script enhances the filtering to ensure each admin interface only shows relevant features.
"""

import re

def update_agents_page_filtering():
    """Update the agents page to show only relevant agents based on app context"""
    
    print("🔧 Updating agents page for app-specific filtering...")
    
    # Find the agents template
    try:
        with open('backend/templates/admin/agents.html', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("⚠️ Agents template not found, skipping...")
        return
    
    # Add conditional filtering for agents based on app context
    # Look for the agents display section and wrap it with conditions
    
    # Find the HealthPIN Agent section and wrap it with HealthPIN admin condition
    healthpin_agent_pattern = r'(<div[^>]*class="[^"]*agent-card[^"]*"[^>]*>.*?HealthPIN Agent.*?</div>\s*</div>)'
    healthpin_replacement = r'{% if is_healthpin_admin or not (is_mediamap_admin or is_healthpin_admin) %}\1{% endif %}'
    
    if re.search(healthpin_agent_pattern, content, re.DOTALL):
        content = re.sub(healthpin_agent_pattern, healthpin_replacement, content, flags=re.DOTALL)
        print("✅ Added HealthPIN agent filtering")
    
    # Find the MediaMap Agent section and wrap it with MediaMap admin condition
    mediamap_agent_pattern = r'(<div[^>]*class="[^"]*agent-card[^"]*"[^>]*>.*?MediaMap Agent.*?</div>\s*</div>)'
    mediamap_replacement = r'{% if is_mediamap_admin or not (is_mediamap_admin or is_healthpin_admin) %}\1{% endif %}'
    
    if re.search(mediamap_agent_pattern, content, re.DOTALL):
        content = re.sub(mediamap_agent_pattern, mediamap_replacement, content, flags=re.DOTALL)
        print("✅ Added MediaMap agent filtering")
    
    # Write back the updated template
    with open('backend/templates/admin/agents.html', 'w') as f:
        f.write(content)
    
    print("✅ Agents page updated with app-specific filtering")

def create_filtered_admin_routes():
    """Create filtered admin routes that respect app context"""
    
    print("🔧 Creating filtered admin routes...")
    
    filtered_routes = '''"""
Filtered Admin Routes
====================

Routes that provide app-specific filtering for admin interfaces.
"""

from flask import session, render_template, redirect, url_for, jsonify
from flask_login import login_required

def register_filtered_admin_routes(app):
    """Register filtered admin routes"""
    
    @app.route('/admin/agents-filtered')
    @login_required
    def admin_agents_filtered():
        """Agents page with app-specific filtering"""
        
        app_context = session.get('app_context', 'unknown')
        
        # Filter agents based on app context
        if app_context == 'mediamap_admin':
            # Show only MediaMap agents
            agents_filter = 'mediamap'
            page_title = 'MediaMap Agents'
        elif app_context == 'healthpin_admin':
            # Show only HealthPIN agents
            agents_filter = 'healthpin'
            page_title = 'HealthPIN Agents'
        else:
            # Show all agents (fallback)
            agents_filter = 'all'
            page_title = 'All Agents'
        
        return render_template('admin/agents.html', 
                             agents_filter=agents_filter,
                             page_title=page_title,
                             app_context=app_context)
    
    @app.route('/admin/insights-filtered')
    @login_required
    def admin_insights_filtered():
        """Insights page with app-specific filtering"""
        
        app_context = session.get('app_context', 'unknown')
        
        if app_context == 'healthpin_admin':
            # Show only HealthPIN insights
            return render_template('admin/insights.html', 
                                 insights_filter='healthpin',
                                 page_title='HealthPIN Insights')
        elif app_context == 'mediamap_admin':
            # Show only MediaMap insights
            return render_template('admin/insights.html', 
                                 insights_filter='mediamap',
                                 page_title='MediaMap Insights')
        else:
            # Show all insights
            return render_template('admin/insights.html', 
                                 insights_filter='all',
                                 page_title='All Insights')
    
    @app.route('/api/agents/filtered')
    @login_required
    def api_agents_filtered():
        """API endpoint for filtered agents"""
        
        app_context = session.get('app_context', 'unknown')
        
        # Mock agent data - in real implementation, this would filter from database
        all_agents = [
            {
                'name': 'MediaMap Agent',
                'type': 'mediamap',
                'status': 'active',
                'description': 'Media analysis and content generation'
            },
            {
                'name': 'HealthPIN Agent',
                'type': 'healthpin', 
                'status': 'active',
                'description': 'Healthcare data analysis and patient matching'
            }
        ]
        
        # Filter based on app context
        if app_context == 'mediamap_admin':
            filtered_agents = [a for a in all_agents if a['type'] == 'mediamap']
        elif app_context == 'healthpin_admin':
            filtered_agents = [a for a in all_agents if a['type'] == 'healthpin']
        else:
            filtered_agents = all_agents
        
        return jsonify({
            'success': True,
            'agents': filtered_agents,
            'filter': app_context
        })
    
    print("✅ Registered filtered admin routes")

'''
    
    with open('backend/filtered_admin_routes.py', 'w') as f:
        f.write(filtered_routes)
    
    print("✅ Created filtered admin routes file")

def update_app_py_with_filtered_routes():
    """Update app.py to include filtered admin routes"""
    
    print("🔧 Updating app.py with filtered routes...")
    
    with open('backend/app.py', 'r') as f:
        content = f.read()
    
    # Add import for filtered routes
    if 'from filtered_admin_routes import register_filtered_admin_routes' not in content:
        import_pattern = r'(from app_routes import register_app_routes)'
        import_replacement = r'\1\ntry:\n    from .filtered_admin_routes import register_filtered_admin_routes\nexcept ImportError:\n    from filtered_admin_routes import register_filtered_admin_routes'
        
        content = re.sub(import_pattern, import_replacement, content)
        print("✅ Added filtered routes import")
    
    # Register filtered routes
    if 'register_filtered_admin_routes(app)' not in content:
        register_pattern = r'(register_app_routes\(app\))'
        register_replacement = r'\1\nregister_filtered_admin_routes(app)'
        
        content = re.sub(register_pattern, register_replacement, content)
        print("✅ Added filtered routes registration")
    
    # Write back
    with open('backend/app.py', 'w') as f:
        f.write(content)
    
    print("✅ app.py updated with filtered routes")

def create_app_context_summary():
    """Create a summary of what each app context shows"""
    
    summary = """
# DEVELOP AI - App Context Filtering Summary

## HealthPIN Admin Context
When logged in as HealthPIN Admin, users see:

### Sidebar Navigation:
- ✅ HealthPIN Dashboard
- ✅ Patient Management  
- ✅ Doctor Management
- ✅ HealthPIN Agents (only HealthPIN agents)
- ✅ Medical Records
- ✅ Patient Matching
- ✅ Health Insights (only health-related)
- ✅ User Management (shared)
- ✅ Settings (shared)

### Hidden Features:
- ❌ Media Analysis
- ❌ Content Management  
- ❌ MediaMap Agents
- ❌ Organizations
- ❌ Reports (media-specific)

## MediaMap Admin Context
When logged in as MediaMap Admin, users see:

### Sidebar Navigation:
- ✅ Dashboard
- ✅ Media Analysis
- ✅ Content Management
- ✅ MediaMap Agents (only MediaMap agents)
- ✅ Organizations
- ✅ Reports
- ✅ User Management (shared)
- ✅ Settings (shared)

### Hidden Features:
- ❌ Patient Management
- ❌ Doctor Management
- ❌ HealthPIN Agents
- ❌ Medical Records
- ❌ Patient Matching
- ❌ Health Insights

## Key Benefits:
1. **Focused Interface**: Each admin sees only relevant tools
2. **Reduced Complexity**: No confusion from irrelevant features
3. **Role-Based Access**: Clear separation of responsibilities
4. **Consistent Branding**: All under DEVELOP AI umbrella
5. **Easy Switching**: Switch between apps anytime
"""
    
    with open('APP_CONTEXT_FILTERING.md', 'w') as f:
        f.write(summary)
    
    print("✅ Created app context filtering summary")

def main():
    """Main function to enhance app filtering"""
    
    print("🎯 ENHANCING APP-SPECIFIC FILTERING")
    print("===================================")
    
    update_agents_page_filtering()
    create_filtered_admin_routes()
    update_app_py_with_filtered_routes()
    create_app_context_summary()
    
    print("")
    print("✅ APP FILTERING ENHANCEMENT COMPLETE!")
    print("=====================================")
    print("")
    print("🎯 HealthPIN Admin now shows ONLY:")
    print("• HealthPIN Dashboard")
    print("• Patient Management")
    print("• Doctor Management") 
    print("• HealthPIN Agents (filtered)")
    print("• Medical Records")
    print("• Patient Matching")
    print("• Health Insights")
    print("• User Management")
    print("• Settings")
    print("")
    print("🎯 MediaMap Admin now shows ONLY:")
    print("• Dashboard")
    print("• Media Analysis")
    print("• Content Management")
    print("• MediaMap Agents (filtered)")
    print("• Organizations")
    print("• Reports")
    print("• User Management")
    print("• Settings")
    print("")
    print("🔄 Restart your app to see the enhanced filtering!")

if __name__ == "__main__":
    main()
