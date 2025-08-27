"""
AIMAP ML API Routes
API endpoints for machine learning predictions and analytics
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required
from typing import Dict, List, Optional
import logging
from ..ml.service import MLService
from ..models import Organisation

logger = logging.getLogger(__name__)

# Create ML API blueprint
ml_api = Blueprint('ml_api', __name__, url_prefix='/api/ml')

# Initialize ML service
ml_service = MLService()

@ml_api.route('/initialize', methods=['POST'])
@login_required
def initialize_models():
    """Initialize and train ML models"""
    try:
        data = request.get_json() or {}
        force_retrain = data.get('force_retrain', False)
        
        ml_service.initialize_models(force_retrain=force_retrain)
        
        return jsonify({
            'status': 'success',
            'message': 'ML models initialized successfully',
            'model_status': ml_service.get_model_status()
        })
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@ml_api.route('/status', methods=['GET'])
@login_required
def get_model_status():
    """Get status of ML models"""
    try:
        status = ml_service.get_model_status()
        return jsonify({
            'status': 'success',
            'data': status
        })
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@ml_api.route('/predict/<int:org_id>', methods=['GET'])
@login_required
def predict_organization(org_id: int):
    """Get comprehensive predictions for an organization"""
    try:
        months_ahead = request.args.get('months_ahead', 12, type=int)
        
        predictions = ml_service.get_organization_predictions(org_id, months_ahead)
        
        return jsonify({
            'status': 'success',
            'data': predictions
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error generating predictions for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to generate predictions: {str(e)}'
        }), 500

@ml_api.route('/roi/<int:org_id>', methods=['POST'])
@login_required
def estimate_roi(org_id: int):
    """Estimate ROI for investment scenario"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Investment scenario data required'
            }), 400
        
        # Validate required fields
        required_fields = ['investment_usd', 'timeline_months']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        roi_analysis = ml_service.estimate_roi_for_investment(org_id, data)
        
        return jsonify({
            'status': 'success',
            'data': roi_analysis
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error estimating ROI for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to estimate ROI: {str(e)}'
        }), 500

@ml_api.route('/sector-insights/<sector>', methods=['GET'])
@login_required
def get_sector_insights(sector: str):
    """Get sector-wide predictive insights"""
    try:
        insights = ml_service.get_sector_insights(sector)
        
        return jsonify({
            'status': 'success',
            'data': insights
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error generating sector insights for {sector}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to generate sector insights: {str(e)}'
        }), 500

@ml_api.route('/recommendations/<int:org_id>', methods=['GET'])
@login_required
def get_investment_recommendations(org_id: int):
    """Get AI investment recommendations for an organization"""
    try:
        recommendations = ml_service.generate_investment_recommendations(org_id)
        
        return jsonify({
            'status': 'success',
            'data': recommendations
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error generating recommendations for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to generate recommendations: {str(e)}'
        }), 500

@ml_api.route('/train', methods=['POST'])
@login_required
def train_models():
    """Force retrain all ML models"""
    try:
        success = ml_service.train_models()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Models trained successfully',
                'model_status': ml_service.get_model_status()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Training failed - insufficient data or other error'
            }), 400
            
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        }), 500

@ml_api.route('/risk-assessment/<int:org_id>', methods=['GET'])
@login_required
def get_risk_assessment(org_id: int):
    """Get detailed risk assessment for an organization"""
    try:
        predictions = ml_service.get_organization_predictions(org_id, months_ahead=6)
        risk_assessment = predictions['risk_assessment']
        
        return jsonify({
            'status': 'success',
            'data': {
                'organization_id': org_id,
                'risk_assessment': risk_assessment,
                'generated_at': predictions['generated_at']
            }
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error getting risk assessment for org {org_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get risk assessment: {str(e)}'
        }), 500

@ml_api.route('/batch-predictions', methods=['POST'])
@login_required
def batch_predictions():
    """Get predictions for multiple organizations"""
    try:
        data = request.get_json()
        org_ids = data.get('organisation_ids', [])
        months_ahead = data.get('months_ahead', 12)
        
        if not org_ids:
            return jsonify({
                'status': 'error',
                'message': 'organisation_ids list required'
            }), 400
        
        results = []
        errors = []
        
        for org_id in org_ids:
            try:
                predictions = ml_service.get_organization_predictions(org_id, months_ahead)
                results.append(predictions)
            except Exception as e:
                errors.append({
                    'organisation_id': org_id,
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': results,
                'errors': errors,
                'successful_count': len(results),
                'error_count': len(errors)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in batch predictions: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Batch prediction failed: {str(e)}'
        }), 500

@ml_api.route('/comparative-analysis', methods=['POST'])
@login_required
def comparative_analysis():
    """Compare multiple organizations with predictions"""
    try:
        data = request.get_json()
        org_ids = data.get('organisation_ids', [])
        
        if len(org_ids) < 2:
            return jsonify({
                'status': 'error',
                'message': 'At least 2 organization IDs required for comparison'
            }), 400
        
        comparisons = []
        
        for org_id in org_ids:
            try:
                predictions = ml_service.get_organization_predictions(org_id, months_ahead=12)
                
                # Extract key metrics for comparison
                org_summary = {
                    'organisation_id': org_id,
                    'name': predictions['organization']['name'],
                    'current_score': predictions['organization']['current_score'],
                    'predicted_score_12m': predictions['trajectory_prediction']['predicted_trajectory'][-1]['predicted_score'],
                    'trend': predictions['trajectory_prediction']['trend'],
                    'risk_level': predictions['risk_assessment']['risk_level'],
                    'risk_score': predictions['risk_assessment']['overall_risk_score'],
                    'peer_percentile': predictions['risk_assessment']['peer_comparison']['percentile_rank']
                }
                
                comparisons.append(org_summary)
                
            except Exception as e:
                logger.warning(f"Failed to get predictions for org {org_id}: {e}")
                continue
        
        # Sort by predicted 12-month score (descending)
        comparisons.sort(key=lambda x: x['predicted_score_12m'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'data': {
                'comparisons': comparisons,
                'analysis': {
                    'top_performer': comparisons[0] if comparisons else None,
                    'highest_risk': max(comparisons, key=lambda x: x['risk_score']) if comparisons else None,
                    'best_improvement': max(comparisons, 
                                         key=lambda x: x['predicted_score_12m'] - x['current_score']) if comparisons else None
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Comparative analysis failed: {str(e)}'
        }), 500
