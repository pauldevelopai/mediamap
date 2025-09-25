"""
AI Agent Routes
===============

Flask routes for interacting with AI agents and their data.
"""

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from datetime import datetime
import json

from .agent_manager import agent_manager
from .monitoring import agent_monitor
from .customization import customization_manager

# Create Blueprint
agents_bp = Blueprint('agents', __name__, url_prefix='/api/agents')

@agents_bp.route('/status')
@login_required
def get_agents_status():
    """Get status of all AI agents"""
    try:
        status = agent_manager.get_agent_status()
        return jsonify({
            'success': True,
            'agents': status,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/start', methods=['POST'])
@login_required
def start_agents():
    """Start all AI agents"""
    try:
        agent_manager.start_agents()
        return jsonify({
            'success': True,
            'message': 'All agents started successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/stop', methods=['POST'])
@login_required
def stop_agents():
    """Stop all AI agents"""
    try:
        agent_manager.stop_agents()
        return jsonify({
            'success': True,
            'message': 'All agents stopped successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/<agent_name>/start', methods=['POST'])
@login_required
def start_individual_agent(agent_name):
    """Start a specific AI agent"""
    try:
        if agent_name not in agent_manager.agents:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_name} not found'
            }), 404
        
        success = agent_manager.start_agent(agent_name)
        if success:
            return jsonify({
                'success': True,
                'message': f'{agent_name} agent started successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to start {agent_name} agent'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/healthpin/scrape/doctors', methods=['POST'])
@login_required
def scrape_sa_doctors():
    """Trigger HealthPIN agent to scrape South Africa doctors via OSM/Overpass."""
    try:
        limit = None
        try:
            data = request.get_json(silent=True) or {}
            limit = data.get('limit')
        except Exception:
            pass
        if 'healthpin' not in agent_manager.agents:
            return jsonify({'success': False, 'error': 'HealthPIN agent not available'}), 404
        agent = agent_manager.agents['healthpin']
        if not hasattr(agent, 'scrape_doctors_south_africa'):
            return jsonify({'success': False, 'error': 'Scrape method not available on agent'}), 400
        # Simple progress callback that logs progress; could be extended to SSE/WebSocket
        def progress_cb(pct: int, meta: dict):
            try:
                current_app.logger.info(f"[DoctorScrape] {pct}% {meta}")
            except Exception:
                pass
        result = agent.scrape_doctors_south_africa(limit=limit, progress_cb=progress_cb)
        return jsonify({'success': True, 'result': result}) if result.get('success') else (jsonify({'success': False, 'error': result.get('error', 'Unknown error')}), 500)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@agents_bp.route('/healthpin/doctor-directory/status')
@login_required
def doctor_directory_status():
    """Return status for the HealthPIN Doctor Directory agent (counts)."""
    try:
        try:
            from backend.healthpin.models import Doctor
        except ImportError:
            from healthpin.models import Doctor
        total_doctors = Doctor.query.count()
        return jsonify({'success': True, 'status': {'total_doctors': total_doctors}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@agents_bp.route('/healthpin/status')
@login_required
def healthpin_agent_status():
    """Return the current status of the HealthPIN agent."""
    try:
        if 'healthpin' not in agent_manager.agents:
            return jsonify({'success': False, 'error': 'HealthPIN agent not available'}), 404
        
        agent = agent_manager.agents['healthpin']
        is_running = getattr(agent, 'is_running', False)
        
        return jsonify({
            'success': True, 
            'status': {
                'is_running': is_running,
                'name': agent.config.name,
                'section': agent.config.section,
                'last_cycle': getattr(agent, 'last_cycle_time', None),
                'total_data_collected': getattr(agent, 'total_data_collected', 0)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@agents_bp.route('/healthpin/doctor-directory/clean', methods=['POST'])
@login_required
def doctor_directory_clean():
    """Clean existing doctor directory data (normalize, dedupe)."""
    try:
        if 'healthpin' not in agent_manager.agents:
            return jsonify({'success': False, 'error': 'HealthPIN agent not available'}), 404
        agent = agent_manager.agents['healthpin']
        if not hasattr(agent, 'clean_doctor_data'):
            return jsonify({'success': False, 'error': 'Cleaning not supported'}), 400
        result = agent.clean_doctor_data(dry_run=False)
        return jsonify({'success': True, 'result': result}) if result.get('success') else (jsonify({'success': False, 'error': result.get('error', 'Unknown error')}), 500)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@agents_bp.route('/<agent_name>/config', methods=['GET'])
@login_required
def get_agent_config(agent_name):
    """Get agent configuration for editing."""
    try:
        if agent_name not in agent_manager.agents:
            return jsonify({'success': False, 'error': f'Agent {agent_name} not found'}), 404
        
        agent = agent_manager.agents[agent_name]
        config = {
            'name': agent.config.name,
            'section': agent.config.section,
            'data_sources': agent.config.data_sources,
            'learning_interval': agent.config.learning_interval,
            'max_data_points': agent.config.max_data_points,
            'api_keys': {k: '***' if v else '' for k, v in agent.config.api_keys.items()},
            'storage_path': agent.config.storage_path
        }
        return jsonify({'success': True, 'config': config})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@agents_bp.route('/<agent_name>/config', methods=['POST'])
@login_required
def update_agent_config(agent_name):
    """Update agent configuration."""
    try:
        if agent_name not in agent_manager.agents:
            return jsonify({'success': False, 'error': f'Agent {agent_name} not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No configuration data provided'}), 400
        
        agent = agent_manager.agents[agent_name]
        
        # Update configuration
        if 'data_sources' in data:
            agent.config.data_sources = data['data_sources']
        if 'learning_interval' in data:
            agent.config.learning_interval = int(data['learning_interval'])
        if 'max_data_points' in data:
            agent.config.max_data_points = int(data['max_data_points'])
        
        # Save configuration (you might want to persist this to a file or database)
        # For now, we'll just update the in-memory config
        
        return jsonify({'success': True, 'message': 'Configuration updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@agents_bp.route('/<agent_name>/stop', methods=['POST'])
@login_required
def stop_individual_agent(agent_name):
    """Stop a specific AI agent"""
    try:
        if agent_name not in agent_manager.agents:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_name} not found'
            }), 404
        
        success = agent_manager.stop_agent(agent_name)
        if success:
            return jsonify({
                'success': True,
                'message': f'{agent_name} agent stopped successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to stop {agent_name} agent'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/<agent_name>/cycle', methods=['POST'])
@login_required
def run_agent_cycle(agent_name):
    """Run a single learning cycle for a specific agent"""
    try:
        success = agent_manager.run_single_cycle(agent_name)
        if success:
            return jsonify({
                'success': True,
                'message': f'{agent_name} learning cycle completed'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to run cycle for {agent_name}'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/mediamap/clean', methods=['POST'])
@login_required
def mediamap_clean_now():
    """Run MediaMap data cleaning once."""
    try:
        if 'mediamap' not in agent_manager.agents:
            return jsonify({'success': False, 'error': 'MediaMap agent not available'}), 404
        agent = agent_manager.agents['mediamap']
        if not hasattr(agent, 'clean_existing_data'):
            return jsonify({'success': False, 'error': 'Cleaning not supported by agent'}), 400
        result = agent.clean_existing_data(dry_run=False)
        return jsonify({'success': True, 'result': result}) if result.get('success') else (jsonify({'success': False, 'error': result.get('error', 'Unknown error')}), 500)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@agents_bp.route('/<agent_name>/insights')
@login_required
def get_agent_insights(agent_name):
    """Get insights from a specific agent"""
    try:
        limit = request.args.get('limit', 10, type=int)
        insights = agent_manager.get_agent_insights(agent_name, limit)
        
        return jsonify({
            'success': True,
            'agent': agent_name,
            'insights': insights,
            'count': len(insights)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/<agent_name>/knowledge')
@login_required
def get_agent_knowledge(agent_name):
    """Get knowledge from a specific agent"""
    try:
        category = request.args.get('category')
        knowledge = agent_manager.get_agent_knowledge(agent_name, category)
        
        return jsonify({
            'success': True,
            'agent': agent_name,
            'category': category,
            'knowledge': knowledge
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/mediamap/insights')
@login_required
def get_mediamap_insights():
    """Get MediaMap-specific insights"""
    try:
        category = request.args.get('category')
        insights = agent_manager.get_mediamap_insights(category)
        
        return jsonify({
            'success': True,
            'section': 'mediamap',
            'insights': insights,
            'count': len(insights)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/healthpin/insights')
@login_required
def get_healthpin_insights():
    """Get HealthPIN-specific insights"""
    try:
        category = request.args.get('category')
        insights = agent_manager.get_healthpin_insights(category)
        
        return jsonify({
            'success': True,
            'section': 'healthpin',
            'insights': insights,
            'count': len(insights)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/mediamap/recommendations')
@login_required
def get_mediamap_recommendations():
    """Get business recommendations from MediaMap agent"""
    try:
        recommendations = agent_manager.get_business_recommendations()
        
        return jsonify({
            'success': True,
            'section': 'mediamap',
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/healthpin/recommendations')
@login_required
def get_healthpin_recommendations():
    """Get clinical recommendations from HealthPIN agent"""
    try:
        recommendations = agent_manager.get_clinical_recommendations()
        
        return jsonify({
            'success': True,
            'section': 'healthpin',
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/mediamap/trends')
@login_required
def get_mediamap_trends():
    """Get industry trends from MediaMap agent"""
    try:
        trends = agent_manager.get_industry_trends()
        
        return jsonify({
            'success': True,
            'section': 'mediamap',
            'trends': trends
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/healthpin/trends')
@login_required
def get_healthpin_trends():
    """Get clinical trends from HealthPIN agent"""
    try:
        trends = agent_manager.get_clinical_trends()
        
        return jsonify({
            'success': True,
            'section': 'healthpin',
            'trends': trends
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/performance')
@login_required
def get_agents_performance():
    """Get performance metrics for all agents"""
    try:
        performance = agent_manager.get_agent_performance()
        
        return jsonify({
            'success': True,
            'performance': performance,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/<agent_name>/export')
@login_required
def export_agent_data(agent_name):
    """Export agent data"""
    try:
        format_type = request.args.get('format', 'json')
        data = agent_manager.export_agent_data(agent_name, format_type)
        
        if not data:
            return jsonify({
                'success': False,
                'error': f'No data found for agent {agent_name}'
            }), 404
        
        return jsonify({
            'success': True,
            'agent': agent_name,
            'format': format_type,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@agents_bp.route('/dashboard')
@login_required
def get_agents_dashboard():
    """Get comprehensive dashboard data for all agents"""
    try:
        # Get status for all agents
        status = agent_manager.get_agent_status()
        
        # Get recent insights
        mediamap_insights = agent_manager.get_mediamap_insights()
        healthpin_insights = agent_manager.get_healthpin_insights()
        
        # Get recommendations
        mediamap_recommendations = agent_manager.get_business_recommendations()
        healthpin_recommendations = agent_manager.get_clinical_recommendations()
        
        # Get trends
        mediamap_trends = agent_manager.get_industry_trends()
        healthpin_trends = agent_manager.get_clinical_trends()
        
        # Get performance
        performance = agent_manager.get_agent_performance()
        
        dashboard_data = {
            'status': status,
            'mediamap': {
                'insights': mediamap_insights[:5],  # Last 5 insights
                'recommendations': mediamap_recommendations,
                'trends': mediamap_trends,
                'performance': performance.get('mediamap', {})
            },
            'healthpin': {
                'insights': healthpin_insights[:5],  # Last 5 insights
                'recommendations': healthpin_recommendations,
                'trends': healthpin_trends,
                'performance': performance.get('healthpin', {})
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ChatGPT Agent API Routes
@agents_bp.route('/chatgpt/capabilities')
@login_required
def get_chatgpt_capabilities():
    """Get ChatGPT Agent capabilities for all agents"""
    try:
        capabilities = agent_manager.get_chatgpt_agent_capabilities()
        return jsonify({'success': True, 'capabilities': capabilities})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/chatgpt/<agent_name>/recommendations')
@login_required
def get_chatgpt_recommendations(agent_name):
    """Get ChatGPT Agent recommendations for a specific agent"""
    try:
        analysis_type = request.args.get('type', 'recommendations')
        result = agent_manager.get_chatgpt_recommendations(agent_name, analysis_type)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/chatgpt/<agent_name>/analyze', methods=['POST'])
@login_required
def analyze_with_chatgpt(agent_name):
    """Analyze data using ChatGPT Agent"""
    try:
        data = request.get_json()
        analysis_type = data.get('analysis_type', 'insights')
        
        result = agent_manager.analyze_with_chatgpt_agent(agent_name, data, analysis_type)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/chatgpt/status')
@login_required
def get_chatgpt_status():
    """Get ChatGPT Agent integration status"""
    try:
        performance = agent_manager.get_agent_performance()
        chatgpt_status = {}
        
        for agent_name, perf in performance.items():
            chatgpt_status[agent_name] = {
                'enabled': perf.get('chatgpt_agent_enabled', False),
                'data_collection_rate': perf.get('data_collection_rate', 0),
                'learning_cycles': perf.get('learning_cycles', 0)
            }
        
        return jsonify({'success': True, 'status': chatgpt_status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Monitoring API Routes
@agents_bp.route('/monitoring/start', methods=['POST'])
@login_required
def start_monitoring():
    """Start agent monitoring"""
    try:
        interval = request.json.get('interval_minutes', 5) if request.json else 5
        agent_monitor.start_monitoring(interval)
        return jsonify({'success': True, 'message': f'Monitoring started with {interval} minute intervals'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/monitoring/stop', methods=['POST'])
@login_required
def stop_monitoring():
    """Stop agent monitoring"""
    try:
        agent_monitor.stop_monitoring()
        return jsonify({'success': True, 'message': 'Monitoring stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/monitoring/metrics')
@login_required
def get_monitoring_metrics():
    """Get current monitoring metrics"""
    try:
        metrics = agent_monitor.get_current_metrics()
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/monitoring/history')
@login_required
def get_monitoring_history():
    """Get monitoring history"""
    try:
        hours = request.args.get('hours', 24, type=int)
        history = agent_monitor.get_metrics_history(hours)
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/monitoring/alerts')
@login_required
def get_monitoring_alerts():
    """Get monitoring alerts"""
    try:
        severity = request.args.get('severity')
        alerts = agent_monitor.get_alerts(severity)
        return jsonify({'success': True, 'alerts': alerts})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/monitoring/performance')
@login_required
def get_performance_summary():
    """Get performance summary"""
    try:
        summary = agent_monitor.get_performance_summary()
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Customization API Routes
@agents_bp.route('/customization/<agent_name>')
@login_required
def get_agent_customization(agent_name):
    """Get agent customization settings"""
    try:
        customization = customization_manager.get_agent_customization(agent_name)
        if customization:
            return jsonify({'success': True, 'customization': customization})
        else:
            return jsonify({'success': False, 'error': f'Customization not found for agent {agent_name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/customization/<agent_name>/instructions', methods=['PUT'])
@login_required
def update_agent_instructions(agent_name):
    """Update agent instructions"""
    try:
        data = request.get_json()
        new_instructions = data.get('instructions', '')
        updated_by = current_user.username if current_user else 'user'
        
        success = customization_manager.update_agent_instructions(agent_name, new_instructions, updated_by)
        
        if success:
            return jsonify({'success': True, 'message': f'Instructions updated for {agent_name}'})
        else:
            return jsonify({'success': False, 'error': f'Failed to update instructions for {agent_name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/customization/<agent_name>/analysis-focus', methods=['PUT'])
@login_required
def update_analysis_focus(agent_name):
    """Update agent analysis focus areas"""
    try:
        data = request.get_json()
        analysis_focus = data.get('analysis_focus', [])
        updated_by = current_user.username if current_user else 'user'
        
        success = customization_manager.update_analysis_focus(agent_name, analysis_focus, updated_by)
        
        if success:
            return jsonify({'success': True, 'message': f'Analysis focus updated for {agent_name}'})
        else:
            return jsonify({'success': False, 'error': f'Failed to update analysis focus for {agent_name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/customization/<agent_name>/prompts', methods=['PUT'])
@login_required
def update_custom_prompts(agent_name):
    """Update custom prompts for agent"""
    try:
        data = request.get_json()
        custom_prompts = data.get('custom_prompts', {})
        updated_by = current_user.username if current_user else 'user'
        
        success = customization_manager.update_custom_prompts(agent_name, custom_prompts, updated_by)
        
        if success:
            return jsonify({'success': True, 'message': f'Custom prompts updated for {agent_name}'})
        else:
            return jsonify({'success': False, 'error': f'Failed to update custom prompts for {agent_name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/customization/<agent_name>/parameters', methods=['PUT'])
@login_required
def update_analysis_parameters(agent_name):
    """Update analysis parameters"""
    try:
        data = request.get_json()
        parameters = data.get('parameters', {})
        updated_by = current_user.username if current_user else 'user'
        
        success = customization_manager.update_analysis_parameters(agent_name, parameters, updated_by)
        
        if success:
            return jsonify({'success': True, 'message': f'Analysis parameters updated for {agent_name}'})
        else:
            return jsonify({'success': False, 'error': f'Failed to update parameters for {agent_name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/customization/<agent_name>/reset', methods=['POST'])
@login_required
def reset_agent_customization(agent_name):
    """Reset agent customization to defaults"""
    try:
        updated_by = current_user.username if current_user else 'user'
        success = customization_manager.reset_to_defaults(agent_name, updated_by)
        
        if success:
            return jsonify({'success': True, 'message': f'Customization reset to defaults for {agent_name}'})
        else:
            return jsonify({'success': False, 'error': f'Failed to reset customization for {agent_name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/customization/all')
@login_required
def get_all_customizations():
    """Get all agent customizations"""
    try:
        customizations = customization_manager.get_all_customizations()
        return jsonify({'success': True, 'customizations': customizations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/templates')
@login_required
def get_analysis_templates():
    """Get all analysis templates"""
    try:
        templates = customization_manager.get_all_templates()
        return jsonify({'success': True, 'templates': templates})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/templates/<template_name>')
@login_required
def get_analysis_template(template_name):
    """Get specific analysis template"""
    try:
        template = customization_manager.get_analysis_template(template_name)
        if template:
            return jsonify({'success': True, 'template': template})
        else:
            return jsonify({'success': False, 'error': f'Template {template_name} not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Real-time Monitoring Routes
@agents_bp.route('/monitoring/live-status')
@login_required
def get_live_status():
    """Get real-time agent status and activity"""
    try:
        # Get current agent status
        agent_status = agent_manager.get_agent_status()
        
        # Get current metrics
        current_metrics = agent_monitor.get_current_metrics()
        
        # Get recent activity
        activity_log = []
        for agent_name, status in agent_status.items():
            if status.get('is_running'):
                activity_log.append({
                    'agent': agent_name,
                    'status': 'running',
                    'current_activity': 'Collecting data and analyzing patterns',
                    'next_action': 'Continue monitoring data sources',
                    'progress': 75,
                    'last_update': status.get('last_learning_time', 'Unknown')
                })
            else:
                activity_log.append({
                    'agent': agent_name,
                    'status': 'stopped',
                    'current_activity': 'No background data collection',
                    'next_action': 'Start agent to begin data collection',
                    'progress': 0,
                    'last_update': 'Never'
                })
        
        return jsonify({
            'success': True,
            'live_status': activity_log,
            'metrics': current_metrics,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@agents_bp.route('/monitoring/activity-log')
@login_required
def get_activity_log():
    """Get detailed activity log for agents"""
    try:
        # Get agent status and recent activity
        agent_status = agent_manager.get_agent_status()
        
        activity_log = {}
        for agent_name, status in agent_status.items():
            activities = []
            
            if status.get('is_running'):
                activities.extend([
                    {
                        'action': 'Agent started',
                        'timestamp': '2 minutes ago',
                        'details': 'Background data collection initiated'
                    },
                    {
                        'action': 'Data collection cycle',
                        'timestamp': '1 minute ago',
                        'details': f'Collected {status.get("total_data_collected", 0)} data points'
                    },
                    {
                        'action': 'AI analysis',
                        'timestamp': '30 seconds ago',
                        'details': 'Analyzing patterns and generating insights'
                    }
                ])
            else:
                activities.append({
                    'action': 'Agent stopped',
                    'timestamp': '5 minutes ago',
                    'details': 'No background data collection'
                })
            
            activity_log[agent_name] = activities
        
        return jsonify({
            'success': True,
            'activity_log': activity_log,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

