"""
AIMAP Consulting API Routes
API endpoints for AI consulting intelligence and strategy generation
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required
from typing import Dict, List, Optional
import logging
from ..consulting.service import AIMAPConsultingService
from backend.aimap.models import Organisation

logger = logging.getLogger(__name__)

# Create consulting API blueprint
consulting_api = Blueprint('consulting_api', __name__, url_prefix='/api/consulting')

# Initialize consulting service
consulting_service = AIMAPConsultingService()

@consulting_api.route('/processes', methods=['GET'])
@login_required
def get_processes():
    """Get available AI processes"""
    try:
        sector = request.args.get('sector', '')
        complexity = request.args.get('complexity', '')
        max_budget = request.args.get('max_budget', type=int)
        
        if sector:
            processes = consulting_service.process_library.get_processes_by_sector(sector)
        elif complexity:
            processes = consulting_service.process_library.get_process_by_complexity(complexity)
        elif max_budget:
            processes = consulting_service.process_library.get_process_by_budget(max_budget)
        else:
            processes = list(consulting_service.process_library.processes.values())
        
        return jsonify({
            'status': 'success',
            'data': {
                'processes': processes,
                'count': len(processes)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting processes: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get processes: {str(e)}'
        }), 500

@consulting_api.route('/strategy/<int:org_id>', methods=['POST'])
@login_required
def generate_strategy(org_id: int):
    """Generate comprehensive AI strategy for an organization"""
    try:
        # Get organization data
        org = Organisation.query.get(org_id)
        if not org:
            return jsonify({
                'status': 'error',
                'message': f'Organization {org_id} not found'
            }), 404
        
        # Get latest metrics
        from backend.aimap.models import Metrics
        latest_metrics = Metrics.query.filter_by(
            organisation_id=org_id
        ).order_by(Metrics.created_at.desc()).first()
        
        # Prepare organization profile
        org_profile = {
            'name': org.name,
            'sector': org.sector,
            'ai_adoption_score': latest_metrics.ai_adoption_score if latest_metrics else 30,
            'team_size': request.json.get('team_size', 5),
            'budget': request.json.get('budget', 100000),
            'ai_tools': org.ai_tools or []
        }
        
        # Generate strategy
        strategy = consulting_service.strategy_generator.generate_strategy(org_profile)
        
        return jsonify({
            'status': 'success',
            'data': strategy
        })
        
    except Exception as e:
        logger.error(f"Error generating strategy for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to generate strategy: {str(e)}'
        }), 500

@consulting_api.route('/package/<int:org_id>', methods=['POST'])
@login_required
def generate_consulting_package(org_id: int):
    """Generate complete consulting package for an organization"""
    try:
        # Get organization data
        org = Organisation.query.get(org_id)
        if not org:
            return jsonify({
                'status': 'error',
                'message': f'Organization {org_id} not found'
            }), 404
        
        # Get latest metrics
        from backend.aimap.models import Metrics
        latest_metrics = Metrics.query.filter_by(
            organisation_id=org_id
        ).order_by(Metrics.created_at.desc()).first()
        
        # Prepare organization profile
        org_profile = {
            'name': org.name,
            'sector': org.sector,
            'ai_adoption_score': latest_metrics.ai_adoption_score if latest_metrics else 30,
            'team_size': request.json.get('team_size', 5),
            'budget': request.json.get('budget', 100000),
            'ai_tools': org.ai_tools or []
        }
        
        # Generate comprehensive package
        package = consulting_service.generate_comprehensive_consulting_package(org_profile)
        
        return jsonify({
            'status': 'success',
            'data': package
        })
        
    except Exception as e:
        logger.error(f"Error generating consulting package for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to generate consulting package: {str(e)}'
        }), 500

@consulting_api.route('/insights/<int:org_id>', methods=['GET'])
@login_required
def get_consulting_insights(org_id: int):
    """Get consulting insights and recommendations for an organization"""
    try:
        # Get organization data
        org = Organisation.query.get(org_id)
        if not org:
            return jsonify({
                'status': 'error',
                'message': f'Organization {org_id} not found'
            }), 404
        
        # Get latest metrics
        from backend.aimap.models import Metrics
        latest_metrics = Metrics.query.filter_by(
            organisation_id=org_id
        ).order_by(Metrics.created_at.desc()).first()
        
        # Prepare organization profile
        org_profile = {
            'name': org.name,
            'sector': org.sector,
            'ai_adoption_score': latest_metrics.ai_adoption_score if latest_metrics else 30,
            'team_size': 5,  # Default
            'budget': 100000,  # Default
            'ai_tools': org.ai_tools or []
        }
        
        # Get insights
        insights = consulting_service.get_consulting_insights(org_profile)
        
        return jsonify({
            'status': 'success',
            'data': insights
        })
        
    except Exception as e:
        logger.error(f"Error getting insights for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get insights: {str(e)}'
        }), 500

@consulting_api.route('/recommendations/<int:org_id>', methods=['GET'])
@login_required
def get_process_recommendations(org_id: int):
    """Get AI process recommendations for an organization"""
    try:
        # Get organization data
        org = Organisation.query.get(org_id)
        if not org:
            return jsonify({
                'status': 'error',
                'message': f'Organization {org_id} not found'
            }), 404
        
        # Get latest metrics
        from backend.aimap.models import Metrics
        latest_metrics = Metrics.query.filter_by(
            organisation_id=org_id
        ).order_by(Metrics.created_at.desc()).first()
        
        # Prepare organization profile
        org_profile = {
            'name': org.name,
            'sector': org.sector,
            'ai_adoption_score': latest_metrics.ai_adoption_score if latest_metrics else 30,
            'team_size': request.args.get('team_size', 5, type=int),
            'budget': request.args.get('budget', 100000, type=int),
            'ai_tools': org.ai_tools or []
        }
        
        # Get recommendations
        recommendations = consulting_service.process_library.get_process_recommendations(org_profile)
        
        return jsonify({
            'status': 'success',
            'data': {
                'organization': org_profile,
                'recommendations': recommendations
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get recommendations: {str(e)}'
        }), 500

@consulting_api.route('/success-plan/<int:org_id>', methods=['POST'])
@login_required
def create_success_plan(org_id: int):
    """Create success tracking plan for an organization"""
    try:
        # Get organization data
        org = Organisation.query.get(org_id)
        if not org:
            return jsonify({
                'status': 'error',
                'message': f'Organization {org_id} not found'
            }), 404
        
        # Get latest metrics
        from backend.aimap.models import Metrics
        latest_metrics = Metrics.query.filter_by(
            organisation_id=org_id
        ).order_by(Metrics.created_at.desc()).first()
        
        # Prepare organization profile
        org_profile = {
            'name': org.name,
            'sector': org.sector,
            'ai_adoption_score': latest_metrics.ai_adoption_score if latest_metrics else 30,
            'team_size': request.json.get('team_size', 5),
            'budget': request.json.get('budget', 100000),
            'ai_tools': org.ai_tools or []
        }
        
        # Generate strategy for success plan
        strategy = consulting_service.strategy_generator.generate_strategy(org_profile)
        
        # Create success plan
        success_plan = consulting_service.success_tracker.create_success_plan(org_profile, strategy)
        
        return jsonify({
            'status': 'success',
            'data': success_plan
        })
        
    except Exception as e:
        logger.error(f"Error creating success plan for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to create success plan: {str(e)}'
        }), 500

@consulting_api.route('/track-progress/<int:org_id>', methods=['POST'])
@login_required
def track_progress(org_id: int):
    """Track progress against success plan"""
    try:
        current_metrics = request.json.get('current_metrics', {})
        
        # Track progress
        progress = consulting_service.success_tracker.track_progress(str(org_id), current_metrics)
        
        return jsonify({
            'status': 'success',
            'data': progress
        })
        
    except Exception as e:
        logger.error(f"Error tracking progress for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to track progress: {str(e)}'
        }), 500

@consulting_api.route('/sectors', methods=['GET'])
@login_required
def get_sectors():
    """Get available sectors for consulting"""
    try:
        sectors = [
            "Media",
            "Communications", 
            "Finance",
            "Healthcare",
            "Manufacturing",
            "Retail",
            "E-commerce",
            "SaaS",
            "Customer Service",
            "Logistics"
        ]
        
        return jsonify({
            'status': 'success',
            'data': {
                'sectors': sectors,
                'count': len(sectors)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting sectors: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get sectors: {str(e)}'
        }), 500

@consulting_api.route('/deliverables/<int:org_id>', methods=['GET'])
@login_required
def get_deliverables(org_id: int):
    """Get consulting deliverables for an organization"""
    try:
        # Get organization data
        org = Organisation.query.get(org_id)
        if not org:
            return jsonify({
                'status': 'error',
                'message': f'Organization {org_id} not found'
            }), 404
        
        # Get latest metrics
        from backend.aimap.models import Metrics
        latest_metrics = Metrics.query.filter_by(
            organisation_id=org_id
        ).order_by(Metrics.created_at.desc()).first()
        
        # Prepare organization profile
        org_profile = {
            'name': org.name,
            'sector': org.sector,
            'ai_adoption_score': latest_metrics.ai_adoption_score if latest_metrics else 30,
            'team_size': 5,
            'budget': 100000,
            'ai_tools': org.ai_tools or []
        }
        
        # Generate package to get deliverables
        package = consulting_service.generate_comprehensive_consulting_package(org_profile)
        deliverables = package.get('deliverables', [])
        
        return jsonify({
            'status': 'success',
            'data': {
                'organization': org_profile,
                'deliverables': deliverables,
                'total_pages': sum(d.get('estimated_pages', 0) for d in deliverables)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting deliverables for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get deliverables: {str(e)}'
        }), 500
