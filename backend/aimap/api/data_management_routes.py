"""
AIMAP Data Management API Routes
API endpoints for managing people, leads, research reports, and custom data
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required
from typing import Dict, List, Optional
import logging
from datetime import datetime, date
from ..models import (
    Organisation, Person, Lead, LeadActivity, Interaction, 
    ResearchReport, CustomData, ConsultingProject, ProjectMilestone
)

logger = logging.getLogger(__name__)

# Create data management API blueprint
data_api = Blueprint('data_api', __name__, url_prefix='/api/data')

@data_api.route('/people', methods=['GET'])
@login_required
def get_people():
    """Get all people/contacts"""
    try:
        org_id = request.args.get('org_id', type=int)
        
        if org_id:
            people = Person.query.filter_by(organisation_id=org_id).all()
        else:
            people = Person.query.all()
        
        return jsonify({
            'status': 'success',
            'data': [{
                'id': p.id,
                'organisation_id': p.organisation_id,
                'first_name': p.first_name,
                'last_name': p.last_name,
                'email': p.email,
                'phone': p.phone,
                'title': p.title,
                'department': p.department,
                'role': p.role,
                'linkedin_url': p.linkedin_url,
                'notes': p.notes,
                'is_primary_contact': p.is_primary_contact,
                'created_at': p.created_at.isoformat() if p.created_at else None
            } for p in people]
        })
        
    except Exception as e:
        logger.error(f"Error getting people: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get people: {str(e)}'
        }), 500

@data_api.route('/people', methods=['POST'])
@login_required
def create_person():
    """Create a new person/contact"""
    try:
        data = request.json
        
        person = Person(
            organisation_id=data['organisation_id'],
            first_name=data['first_name'],
            last_name=data['last_name'],
            email=data.get('email'),
            phone=data.get('phone'),
            title=data.get('title'),
            department=data.get('department'),
            role=data.get('role'),
            linkedin_url=data.get('linkedin_url'),
            notes=data.get('notes'),
            is_primary_contact=data.get('is_primary_contact', False)
        )
        
        from ..db import db
        db.session.add(person)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'data': {
                'id': person.id,
                'message': 'Person created successfully'
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating person: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to create person: {str(e)}'
        }), 500

@data_api.route('/leads', methods=['GET'])
@login_required
def get_leads():
    """Get all leads/prospects"""
    try:
        org_id = request.args.get('org_id', type=int)
        status = request.args.get('status')
        
        query = Lead.query
        
        if org_id:
            query = query.filter_by(organisation_id=org_id)
        if status:
            query = query.filter_by(status=status)
        
        leads = query.all()
        
        return jsonify({
            'status': 'success',
            'data': [{
                'id': l.id,
                'organisation_id': l.organisation_id,
                'status': l.status,
                'source': l.source,
                'priority': l.priority,
                'estimated_value': l.estimated_value,
                'probability': l.probability,
                'expected_close_date': l.expected_close_date.isoformat() if l.expected_close_date else None,
                'assigned_to': l.assigned_to,
                'notes': l.notes,
                'created_at': l.created_at.isoformat() if l.created_at else None
            } for l in leads]
        })
        
    except Exception as e:
        logger.error(f"Error getting leads: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get leads: {str(e)}'
        }), 500

@data_api.route('/leads', methods=['POST'])
@login_required
def create_lead():
    """Create a new lead/prospect"""
    try:
        data = request.json
        
        lead = Lead(
            organisation_id=data['organisation_id'],
            status=data.get('status', 'Prospect'),
            source=data.get('source'),
            priority=data.get('priority', 'Medium'),
            estimated_value=data.get('estimated_value'),
            probability=data.get('probability', 0.0),
            expected_close_date=datetime.strptime(data['expected_close_date'], '%Y-%m-%d').date() if data.get('expected_close_date') else None,
            assigned_to=data.get('assigned_to'),
            notes=data.get('notes')
        )
        
        from ..db import db
        db.session.add(lead)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'data': {
                'id': lead.id,
                'message': 'Lead created successfully'
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating lead: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to create lead: {str(e)}'
        }), 500

@data_api.route('/leads/<int:lead_id>/activities', methods=['POST'])
@login_required
def add_lead_activity(lead_id: int):
    """Add activity to a lead"""
    try:
        data = request.json
        
        activity = LeadActivity(
            lead_id=lead_id,
            activity_type=data['activity_type'],
            description=data['description'],
            outcome=data.get('outcome'),
            next_action=data.get('next_action'),
            scheduled_date=datetime.strptime(data['scheduled_date'], '%Y-%m-%d %H:%M:%S') if data.get('scheduled_date') else None,
            completed_date=datetime.strptime(data['completed_date'], '%Y-%m-%d %H:%M:%S') if data.get('completed_date') else None
        )
        
        from ..db import db
        db.session.add(activity)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'data': {
                'id': activity.id,
                'message': 'Activity added successfully'
            }
        })
        
    except Exception as e:
        logger.error(f"Error adding lead activity: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to add activity: {str(e)}'
        }), 500

@data_api.route('/research', methods=['GET'])
@login_required
def get_research_reports():
    """Get all research reports"""
    try:
        org_id = request.args.get('org_id', type=int)
        report_type = request.args.get('report_type')
        
        query = ResearchReport.query
        
        if org_id:
            query = query.filter_by(organisation_id=org_id)
        if report_type:
            query = query.filter_by(report_type=report_type)
        
        reports = query.all()
        
        return jsonify({
            'status': 'success',
            'data': [{
                'id': r.id,
                'organisation_id': r.organisation_id,
                'title': r.title,
                'description': r.description,
                'report_type': r.report_type,
                'file_path': r.file_path,
                'file_size': r.file_size,
                'file_type': r.file_type,
                'tags': r.tags or [],
                'ai_insights': r.ai_insights or {},
                'summary': r.summary,
                'author': r.author,
                'publication_date': r.publication_date.isoformat() if r.publication_date else None,
                'created_at': r.created_at.isoformat() if r.created_at else None
            } for r in reports]
        })
        
    except Exception as e:
        logger.error(f"Error getting research reports: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get research reports: {str(e)}'
        }), 500

@data_api.route('/research', methods=['POST'])
@login_required
def create_research_report():
    """Create a new research report"""
    try:
        data = request.json
        
        report = ResearchReport(
            organisation_id=data.get('organisation_id'),
            title=data['title'],
            description=data.get('description'),
            report_type=data.get('report_type'),
            file_path=data.get('file_path'),
            file_size=data.get('file_size'),
            file_type=data.get('file_type'),
            tags=data.get('tags', []),
            ai_insights=data.get('ai_insights', {}),
            summary=data.get('summary'),
            author=data.get('author'),
            publication_date=datetime.strptime(data['publication_date'], '%Y-%m-%d').date() if data.get('publication_date') else None
        )
        
        from ..db import db
        db.session.add(report)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'data': {
                'id': report.id,
                'message': 'Research report created successfully'
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating research report: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to create research report: {str(e)}'
        }), 500

@data_api.route('/custom', methods=['GET'])
@login_required
def get_custom_data():
    """Get custom data by type"""
    try:
        data_type = request.args.get('data_type')
        
        query = CustomData.query
        
        if data_type:
            query = query.filter_by(data_type=data_type)
        
        custom_data = query.all()
        
        return jsonify({
            'status': 'success',
            'data': [{
                'id': c.id,
                'data_type': c.data_type,
                'title': c.title,
                'content': c.content or {},
                'tags': c.tags or [],
                'metadata': c.metadata or {},
                'created_at': c.created_at.isoformat() if c.created_at else None
            } for c in custom_data]
        })
        
    except Exception as e:
        logger.error(f"Error getting custom data: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get custom data: {str(e)}'
        }), 500

@data_api.route('/custom', methods=['POST'])
@login_required
def create_custom_data():
    """Create custom data"""
    try:
        data = request.json
        
        custom_data = CustomData(
            data_type=data['data_type'],
            title=data['title'],
            content=data.get('content', {}),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
        
        from ..db import db
        db.session.add(custom_data)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'data': {
                'id': custom_data.id,
                'message': 'Custom data created successfully'
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating custom data: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to create custom data: {str(e)}'
        }), 500

@data_api.route('/projects', methods=['GET'])
@login_required
def get_consulting_projects():
    """Get consulting projects"""
    try:
        org_id = request.args.get('org_id', type=int)
        status = request.args.get('status')
        
        query = ConsultingProject.query
        
        if org_id:
            query = query.filter_by(organisation_id=org_id)
        if status:
            query = query.filter_by(status=status)
        
        projects = query.all()
        
        return jsonify({
            'status': 'success',
            'data': [{
                'id': p.id,
                'organisation_id': p.organisation_id,
                'project_name': p.project_name,
                'project_type': p.project_type,
                'status': p.status,
                'start_date': p.start_date.isoformat() if p.start_date else None,
                'end_date': p.end_date.isoformat() if p.end_date else None,
                'budget': p.budget,
                'actual_cost': p.actual_cost,
                'description': p.description,
                'objectives': p.objectives or [],
                'deliverables': p.deliverables or [],
                'team_members': p.team_members or [],
                'notes': p.notes,
                'created_at': p.created_at.isoformat() if p.created_at else None
            } for p in projects]
        })
        
    except Exception as e:
        logger.error(f"Error getting consulting projects: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get consulting projects: {str(e)}'
        }), 500

@data_api.route('/projects', methods=['POST'])
@login_required
def create_consulting_project():
    """Create a new consulting project"""
    try:
        data = request.json
        
        project = ConsultingProject(
            organisation_id=data['organisation_id'],
            project_name=data['project_name'],
            project_type=data.get('project_type'),
            status=data.get('status', 'Planning'),
            start_date=datetime.strptime(data['start_date'], '%Y-%m-%d').date() if data.get('start_date') else None,
            end_date=datetime.strptime(data['end_date'], '%Y-%m-%d').date() if data.get('end_date') else None,
            budget=data.get('budget'),
            actual_cost=data.get('actual_cost'),
            description=data.get('description'),
            objectives=data.get('objectives', []),
            deliverables=data.get('deliverables', []),
            team_members=data.get('team_members', []),
            notes=data.get('notes')
        )
        
        from ..db import db
        db.session.add(project)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'data': {
                'id': project.id,
                'message': 'Consulting project created successfully'
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating consulting project: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to create consulting project: {str(e)}'
        }), 500

@data_api.route('/data-types', methods=['GET'])
@login_required
def get_data_types():
    """Get available data types for custom data"""
    try:
        data_types = [
            'competitor',
            'market_trend', 
            'tool_review',
            'case_study',
            'industry_analysis',
            'technology_assessment',
            'vendor_evaluation',
            'best_practice',
            'regulatory_update',
            'event_notes'
        ]
        
        return jsonify({
            'status': 'success',
            'data': {
                'data_types': data_types,
                'count': len(data_types)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting data types: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get data types: {str(e)}'
        }), 500
