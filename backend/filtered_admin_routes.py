"""
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

