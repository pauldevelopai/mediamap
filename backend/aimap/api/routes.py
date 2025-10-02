"""
AIMAP API Routes
Flask blueprints for AIMAP API endpoints
"""
from flask import Blueprint, request, jsonify, send_file
from flask_login import login_required
from typing import Dict, List, Optional
import os
from backend.aimap.models import Organisation, Metrics, db
from ..ingest.pipeline import IngestionPipeline
from ..scoring.engine import ScoringEngine
from ..reports.pptx_export import PPTXReportGenerator
from ..reports.pdf_export import PDFReportGenerator
from ..config import DEFAULT_PERIOD

# Create API blueprint
aimap_api = Blueprint('aimap_api', __name__, url_prefix='/api')

# Initialize services
ingestion_pipeline = IngestionPipeline()
scoring_engine = ScoringEngine()
pptx_generator = PPTXReportGenerator()
pdf_generator = PDFReportGenerator()

@aimap_api.route('/organisations', methods=['GET'])
@login_required
def get_organisations():
    """Get organisations with optional filtering"""
    try:
        # Get query parameters
        sector = request.args.get('sector')
        region = request.args.get('region')
        country = request.args.get('country')
        size_band = request.args.get('size_band')
        client_tag = request.args.get('client_tag')
        
        # Build query
        query = Organisation.query
        
        if sector:
            query = query.filter(Organisation.sector == sector)
        if region:
            query = query.filter(Organisation.region == region)
        if country:
            query = query.filter(Organisation.country == country)
        if size_band:
            query = query.filter(Organisation.size_band == size_band)
        if client_tag:
            query = query.filter(Organisation.client_tag == client_tag)
        
        organisations = query.all()
        
        # Convert to dict and add latest metrics
        result = []
        for org in organisations:
            org_data = org.to_dict()
            
            # Get latest metrics
            latest_metric = Metrics.query.filter_by(
                organisation_id=org.id
            ).order_by(Metrics.created_at.desc()).first()
            
            if latest_metric:
                org_data['latest_metrics'] = latest_metric.to_dict()
            else:
                org_data['latest_metrics'] = None
            
            result.append(org_data)
        
        return jsonify({
            'status': 'success',
            'data': result,
            'count': len(result)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@aimap_api.route('/organisations/<int:org_id>', methods=['GET'])
@login_required
def get_organisation(org_id: int):
    """Get detailed organisation information"""
    try:
        org = Organisation.query.get_or_404(org_id)
        
        # Get all metrics for this organisation
        metrics = Metrics.query.filter_by(
            organisation_id=org_id
        ).order_by(Metrics.period.desc()).all()
        
        org_data = org.to_dict()
        org_data['metrics_history'] = [m.to_dict() for m in metrics]
        
        # Get latest metrics
        if metrics:
            org_data['latest_metrics'] = metrics[0].to_dict()
            
            # Get peer benchmarks for latest period
            latest_metric = metrics[0]
            if latest_metric.benchmark_bucket:
                benchmark_data = scoring_engine.get_peer_benchmarks(
                    latest_metric.benchmark_bucket, 
                    latest_metric.period
                )
                org_data['peer_benchmarks'] = benchmark_data
        else:
            org_data['latest_metrics'] = None
            org_data['peer_benchmarks'] = None
        
        return jsonify({
            'status': 'success',
            'data': org_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@aimap_api.route('/organisations', methods=['POST'])
@login_required
def create_organisation():
    """Create new organisation"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('name'):
            return jsonify({
                'status': 'error',
                'message': 'Name is required'
            }), 400
        
        # Check if organisation already exists
        existing = Organisation.query.filter_by(name=data['name']).first()
        if existing:
            return jsonify({
                'status': 'error',
                'message': 'Organisation with this name already exists'
            }), 400
        
        # Create organisation
        org = Organisation(
            name=data['name'],
            sector=data.get('sector', 'Media'),
            subsector=data.get('subsector'),
            region=data.get('region'),
            country=data.get('country'),
            size_band=data.get('size_band'),
            client_tag=data.get('client_tag'),
            contact=data.get('contact'),
            website_url=data.get('website_url'),
            notes=data.get('notes')
        )
        
        db.session.add(org)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'data': org.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@aimap_api.route('/ingest/run', methods=['POST'])
@login_required
def run_ingestion():
    """Run ingestion pipeline"""
    try:
        data = request.get_json() or {}
        
        sector = data.get('sector')
        organisation = data.get('organisation')
        dry_run = data.get('dry_run', False)
        period = data.get('period', DEFAULT_PERIOD)
        
        if organisation:
            # Process single organisation
            result = ingestion_pipeline.process_by_name(organisation, period, dry_run)
            results = [result]
        elif sector:
            # Process all organisations in sector
            results = ingestion_pipeline.process_sector(sector, period, dry_run)
        else:
            # Process all organisations
            results = ingestion_pipeline.process_all(period, dry_run)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'processed': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@aimap_api.route('/score/run', methods=['POST'])
@login_required
def run_scoring():
    """Run scoring for organisations"""
    try:
        data = request.get_json() or {}
        
        period = data.get('period', DEFAULT_PERIOD)
        sector = data.get('sector')
        organisation = data.get('organisation')
        
        if organisation:
            # Score single organisation
            org = Organisation.query.filter_by(name=organisation).first()
            if not org:
                return jsonify({
                    'status': 'error',
                    'message': f'Organisation "{organisation}" not found'
                }), 404
            
            # Get existing metrics or create new
            metric = Metrics.query.filter_by(
                organisation_id=org.id,
                period=period
            ).first()
            
            if not metric:
                return jsonify({
                    'status': 'error',
                    'message': f'No signals data found for {organisation} in period {period}'
                }), 404
            
            # Re-score based on existing signals
            score, maturity_stage = scoring_engine.score_organisation(org, metric.signals or {}, period)
            benchmark_bucket = scoring_engine.create_benchmark_bucket(org)
            
            metric.ai_adoption_score = score
            metric.maturity_stage = maturity_stage
            metric.benchmark_bucket = benchmark_bucket
            
            db.session.commit()
            
            results = [{
                'organisation': org.name,
                'score': score,
                'maturity_stage': maturity_stage,
                'benchmark_bucket': benchmark_bucket
            }]
        else:
            # Score multiple organisations
            query = Organisation.query
            if sector:
                query = query.filter(Organisation.sector == sector)
            
            orgs = query.all()
            results = []
            
            for org in orgs:
                metric = Metrics.query.filter_by(
                    organisation_id=org.id,
                    period=period
                ).first()
                
                if metric and metric.signals:
                    score, maturity_stage = scoring_engine.score_organisation(org, metric.signals, period)
                    benchmark_bucket = scoring_engine.create_benchmark_bucket(org)
                    
                    metric.ai_adoption_score = score
                    metric.maturity_stage = maturity_stage
                    metric.benchmark_bucket = benchmark_bucket
                    
                    results.append({
                        'organisation': org.name,
                        'score': score,
                        'maturity_stage': maturity_stage,
                        'benchmark_bucket': benchmark_bucket
                    })
            
            db.session.commit()
        
        return jsonify({
            'status': 'success',
            'results': results,
            'processed': len(results)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@aimap_api.route('/reports/<int:org_id>/pptx', methods=['POST'])
@login_required
def generate_pptx_report(org_id: int):
    """Generate PPTX report for organisation"""
    try:
        data = request.get_json() or {}
        period = data.get('period', DEFAULT_PERIOD)
        logo_path = data.get('logo_path')
        
        org = Organisation.query.get_or_404(org_id)
        
        # Generate report
        report_path = pptx_generator.generate_report(org, period, logo_path)
        
        # Return file
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"{org.name}_{period}_report.pptx",
            mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation'
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@aimap_api.route('/reports/<int:org_id>/pdf', methods=['POST'])
@login_required
def generate_pdf_report(org_id: int):
    """Generate PDF report for organisation"""
    try:
        data = request.get_json() or {}
        period = data.get('period', DEFAULT_PERIOD)
        logo_path = data.get('logo_path')
        
        org = Organisation.query.get_or_404(org_id)
        
        # Generate report
        report_path = pdf_generator.generate_report(org, period, logo_path)
        
        # Return file
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"{org.name}_{period}_report.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@aimap_api.route('/benchmarks', methods=['GET'])
@login_required
def get_benchmarks():
    """Get benchmark data for different sectors/regions"""
    try:
        sector = request.args.get('sector')
        region = request.args.get('region')
        period = request.args.get('period', DEFAULT_PERIOD)
        
        # Get all metrics for the period
        query = Metrics.query.filter_by(period=period)
        
        if sector:
            # Filter by sector through organisation join
            query = query.join(Organisation).filter(Organisation.sector == sector)
        
        metrics = query.all()
        
        # Group by benchmark bucket
        benchmark_data = {}
        for metric in metrics:
            if not metric.benchmark_bucket or metric.ai_adoption_score is None:
                continue
            
            bucket = metric.benchmark_bucket
            if bucket not in benchmark_data:
                benchmark_data[bucket] = []
            
            benchmark_data[bucket].append(metric.ai_adoption_score)
        
        # Calculate statistics for each bucket
        results = []
        for bucket, scores in benchmark_data.items():
            if len(scores) >= 2:  # Need at least 2 data points
                import statistics
                results.append({
                    'bucket': bucket,
                    'median_score': statistics.median(scores),
                    'p25_score': statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else min(scores),
                    'p75_score': statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else max(scores),
                    'count': len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores)
                })
        
        return jsonify({
            'status': 'success',
            'data': results,
            'period': period
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
