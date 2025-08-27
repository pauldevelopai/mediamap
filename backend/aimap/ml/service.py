"""
AIMAP ML Service
Main service for managing ML models and predictions
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from .models import AIAdoptionPredictor, RiskScorer, ROIEstimator
from .data_generator import TrainingDataGenerator
from ..models import Organisation, Metrics, db

logger = logging.getLogger(__name__)

class MLService:
    """Main service for AIMAP machine learning functionality"""
    
    def __init__(self):
        self.adoption_predictor = AIAdoptionPredictor()
        self.risk_scorer = RiskScorer()
        self.roi_estimator = ROIEstimator()
        self.data_generator = TrainingDataGenerator()
        self._models_trained = False
    
    def initialize_models(self, force_retrain: bool = False):
        """Initialize and train ML models if needed"""
        logger.info("Initializing AIMAP ML models...")
        
        # Try to load existing models
        try:
            self.adoption_predictor.load_model()
            self._models_trained = True
            logger.info("Loaded existing trained models")
        except:
            logger.info("No existing models found, training new models...")
            force_retrain = True
        
        if force_retrain or not self._models_trained:
            self.train_models()
    
    def train_models(self):
        """Train all ML models with current data"""
        logger.info("Training ML models...")
        
        # Generate training dataset
        training_df = self.data_generator.create_training_dataset(min_orgs=100)
        
        if len(training_df) < 50:
            logger.warning("Insufficient training data. Need at least 50 data points.")
            return False
        
        # Prepare training data for adoption predictor
        training_data = []
        target_scores = []
        
        for _, row in training_df.iterrows():
            org_data = {
                'current_score': row['ai_adoption_score'],
                'sector': row['sector'],
                'size_band': row['size_band'],
                'region': row['region'],
                'ai_tools': row['ai_tools'],
                'months_active': row.get('months_active', 12),
                'score_change_3m': row.get('score_change_3m', 0),
                'peer_ranking': row.get('peer_ranking', 50)
            }
            
            # For training, use next month's score as target (if available)
            # For now, simulate target by adding some realistic improvement
            target_score = min(100, row['ai_adoption_score'] + abs(row.get('score_velocity', 1)))
            
            training_data.append(org_data)
            target_scores.append(target_score)
        
        # Train adoption predictor
        try:
            mae = self.adoption_predictor.train(training_data, target_scores)
            logger.info(f"Adoption predictor trained successfully (MAE: {mae:.2f})")
            self._models_trained = True
        except Exception as e:
            logger.error(f"Failed to train adoption predictor: {e}")
            return False
        
        return True
    
    def get_organization_predictions(self, org_id: int, months_ahead: int = 12) -> Dict:
        """Get comprehensive predictions for an organization"""
        
        if not self._models_trained:
            self.initialize_models()
        
        # Get organization data
        org = Organisation.query.get(org_id)
        if not org:
            raise ValueError(f"Organization {org_id} not found")
        
        # Get latest metrics
        latest_metrics = Metrics.query.filter_by(
            organisation_id=org_id
        ).order_by(Metrics.created_at.desc()).first()
        
        if not latest_metrics:
            raise ValueError(f"No metrics found for organization {org_id}")
        
        # Prepare organization data
        org_data = {
            'current_score': latest_metrics.ai_adoption_score or 0,
            'sector': org.sector,
            'size_band': org.size_band or 'medium',
            'region': org.region or 'Unknown',
            'ai_tools': org.ai_tools or [],
            'months_active': 12,  # Default assumption
            'score_change_3m': 0,  # Would be calculated from historical data
            'peer_ranking': 50     # Would be calculated from peer comparison
        }
        
        # Get predictions
        trajectory = self.adoption_predictor.predict_trajectory(org_data, months_ahead)
        
        # Get peer data for risk assessment
        peer_data = self.data_generator.generate_peer_data(org)
        risk_assessment = self.risk_scorer.calculate_risk_score(
            {**org_data, 'ai_adoption_score': org_data['current_score']}, 
            peer_data
        )
        
        return {
            'organization': {
                'id': org.id,
                'name': org.name,
                'sector': org.sector,
                'current_score': org_data['current_score']
            },
            'trajectory_prediction': trajectory,
            'risk_assessment': risk_assessment,
            'generated_at': datetime.now().isoformat()
        }
    
    def estimate_roi_for_investment(self, org_id: int, investment_scenario: Dict) -> Dict:
        """Estimate ROI for a specific investment scenario"""
        
        # Get organization data
        org = Organisation.query.get(org_id)
        if not org:
            raise ValueError(f"Organization {org_id} not found")
        
        latest_metrics = Metrics.query.filter_by(
            organisation_id=org_id
        ).order_by(Metrics.created_at.desc()).first()
        
        org_data = {
            'sector': org.sector,
            'size_band': org.size_band or 'medium',
            'ai_adoption_score': latest_metrics.ai_adoption_score if latest_metrics else 30,
            'region': org.region or 'Unknown'
        }
        
        # Calculate ROI
        roi_analysis = self.roi_estimator.estimate_roi(org_data, investment_scenario)
        
        return {
            'organization': {
                'id': org.id,
                'name': org.name,
                'sector': org.sector
            },
            'investment_scenario': investment_scenario,
            'roi_analysis': roi_analysis,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_sector_insights(self, sector: str) -> Dict:
        """Get sector-wide predictive insights"""
        
        # Get all organizations in sector
        orgs = Organisation.query.filter_by(sector=sector).all()
        
        if not orgs:
            raise ValueError(f"No organizations found in sector: {sector}")
        
        # Collect predictions for all organizations
        sector_predictions = []
        risk_levels = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        total_current_score = 0
        
        for org in orgs:
            try:
                predictions = self.get_organization_predictions(org.id, months_ahead=6)
                sector_predictions.append(predictions)
                
                # Aggregate risk levels
                risk_level = predictions['risk_assessment']['risk_level']
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
                
                total_current_score += predictions['organization']['current_score']
                
            except Exception as e:
                logger.warning(f"Failed to get predictions for org {org.id}: {e}")
                continue
        
        if not sector_predictions:
            raise ValueError(f"Could not generate predictions for any organization in {sector}")
        
        # Calculate sector averages and trends
        avg_current_score = total_current_score / len(sector_predictions)
        
        # Predict sector average in 6 months
        future_scores = []
        for pred in sector_predictions:
            if pred['trajectory_prediction']['predicted_trajectory']:
                # Get 6-month prediction
                six_month_pred = pred['trajectory_prediction']['predicted_trajectory'][5]  # Month 6
                future_scores.append(six_month_pred['predicted_score'])
        
        avg_future_score = sum(future_scores) / len(future_scores) if future_scores else avg_current_score
        
        return {
            'sector': sector,
            'organization_count': len(orgs),
            'predictions_generated': len(sector_predictions),
            'current_metrics': {
                'average_score': round(avg_current_score, 1),
                'risk_distribution': risk_levels
            },
            'six_month_forecast': {
                'predicted_average_score': round(avg_future_score, 1),
                'expected_improvement': round(avg_future_score - avg_current_score, 1)
            },
            'sector_trend': 'improving' if avg_future_score > avg_current_score else 'declining',
            'generated_at': datetime.now().isoformat()
        }
    
    def get_model_status(self) -> Dict:
        """Get status of all ML models"""
        return {
            'models_trained': self._models_trained,
            'adoption_predictor': {
                'available': self.adoption_predictor.model is not None,
                'model_type': 'XGBoost Regressor'
            },
            'risk_scorer': {
                'available': True,
                'model_type': 'Rule-based + Gradient Boosting'
            },
            'roi_estimator': {
                'available': True,
                'model_type': 'Industry benchmark-based'
            }
        }
    
    def generate_investment_recommendations(self, org_id: int) -> Dict:
        """Generate AI investment recommendations for an organization"""
        
        # Get current predictions
        predictions = self.get_organization_predictions(org_id)
        risk_level = predictions['risk_assessment']['risk_level']
        current_score = predictions['organization']['current_score']
        
        # Define investment scenarios based on risk level and current score
        if risk_level in ['high', 'critical'] or current_score < 30:
            # Aggressive investment needed
            scenarios = [
                {
                    'name': 'Emergency AI Transformation',
                    'investment_usd': 200000,
                    'timeline_months': 6,
                    'target_score': min(100, current_score + 40),
                    'description': 'Comprehensive AI adoption program with external consulting'
                },
                {
                    'name': 'Rapid Pilot Implementation',
                    'investment_usd': 100000,
                    'timeline_months': 4,
                    'target_score': min(100, current_score + 25),
                    'description': 'Fast-track pilot projects with dedicated team'
                }
            ]
        elif current_score < 60:
            # Moderate investment
            scenarios = [
                {
                    'name': 'Structured AI Expansion',
                    'investment_usd': 150000,
                    'timeline_months': 12,
                    'target_score': min(100, current_score + 30),
                    'description': 'Systematic expansion of AI capabilities'
                },
                {
                    'name': 'Incremental Growth',
                    'investment_usd': 75000,
                    'timeline_months': 8,
                    'target_score': min(100, current_score + 20),
                    'description': 'Gradual AI adoption with internal resources'
                }
            ]
        else:
            # Optimization investment
            scenarios = [
                {
                    'name': 'AI Excellence Program',
                    'investment_usd': 100000,
                    'timeline_months': 12,
                    'target_score': min(100, current_score + 15),
                    'description': 'Advanced AI optimization and innovation'
                },
                {
                    'name': 'Continuous Improvement',
                    'investment_usd': 50000,
                    'timeline_months': 6,
                    'target_score': min(100, current_score + 10),
                    'description': 'Fine-tuning and advanced capabilities'
                }
            ]
        
        # Calculate ROI for each scenario
        recommendations = []
        for scenario in scenarios:
            roi_analysis = self.estimate_roi_for_investment(org_id, scenario)
            recommendations.append({
                'scenario': scenario,
                'roi_analysis': roi_analysis['roi_analysis']
            })
        
        return {
            'organization_id': org_id,
            'risk_level': risk_level,
            'current_score': current_score,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
