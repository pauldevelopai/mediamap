"""
AIMAP ML Models
Predictive models for AI adoption forecasting, risk assessment, and ROI estimation
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, classification_report
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime, timedelta
from ..models import Organisation, Metrics
from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)

class AIAdoptionPredictor:
    """Predicts future AI adoption trajectories for organizations"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'current_score', 'sector_encoded', 'size_band_encoded', 'region_encoded',
            'tools_count', 'months_since_start', 'score_velocity', 'peer_percentile'
        ]
        self.model_path = PROJECT_ROOT / "backend" / "aimap" / "ml" / "saved_models"
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_features(self, org_data: List[Dict]) -> pd.DataFrame:
        """Prepare features for prediction model"""
        df = pd.DataFrame(org_data)
        
        # Encode categorical variables
        le_sector = LabelEncoder()
        le_size = LabelEncoder()
        le_region = LabelEncoder()
        
        df['sector_encoded'] = le_sector.fit_transform(df['sector'].fillna('Unknown'))
        df['size_band_encoded'] = le_size.fit_transform(df['size_band'].fillna('Unknown'))
        df['region_encoded'] = le_region.fit_transform(df['region'].fillna('Unknown'))
        
        # Calculate derived features
        df['tools_count'] = df['ai_tools'].apply(lambda x: len(x) if x else 0)
        df['months_since_start'] = df.get('months_active', 12)  # Default to 12 months
        df['score_velocity'] = df.get('score_change_3m', 0)  # 3-month score change
        df['peer_percentile'] = df.get('peer_ranking', 50)  # Default to median
        
        return df[self.feature_columns]
    
    def train(self, training_data: List[Dict], target_scores: List[float]):
        """Train the adoption prediction model"""
        logger.info("Training AI adoption prediction model...")
        
        X = self.prepare_features(training_data)
        y = np.array(target_scores)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        logger.info(f"Model trained with MAE: {mae:.2f}")
        
        # Save model
        self.save_model()
        
        return mae
    
    def predict_trajectory(self, org_data: Dict, months_ahead: int = 12) -> Dict:
        """Predict AI adoption trajectory for an organization"""
        if self.model is None:
            self.load_model()
        
        # Prepare current features
        features_df = self.prepare_features([org_data])
        features_scaled = self.scaler.transform(features_df)
        
        # Generate predictions for future months
        trajectory = []
        current_score = org_data.get('current_score', 0)
        
        for month in range(1, months_ahead + 1):
            # Adjust features for future prediction
            future_features = features_scaled.copy()
            future_features[0][5] += month  # months_since_start
            
            predicted_score = self.model.predict(future_features)[0]
            
            # Apply constraints (scores should be realistic)
            predicted_score = max(0, min(100, predicted_score))
            
            trajectory.append({
                'month': month,
                'predicted_score': predicted_score,
                'confidence_interval': self._calculate_confidence(predicted_score)
            })
        
        return {
            'current_score': current_score,
            'predicted_trajectory': trajectory,
            'trend': self._analyze_trend(trajectory),
            'peak_score_month': self._find_peak_month(trajectory)
        }
    
    def _calculate_confidence(self, score: float) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Simple confidence interval based on model uncertainty
        margin = 5.0  # ±5 points
        return (max(0, score - margin), min(100, score + margin))
    
    def _analyze_trend(self, trajectory: List[Dict]) -> str:
        """Analyze overall trend in trajectory"""
        scores = [point['predicted_score'] for point in trajectory]
        if len(scores) < 2:
            return 'insufficient_data'
        
        start_score = scores[0]
        end_score = scores[-1]
        
        if end_score > start_score + 10:
            return 'accelerating'
        elif end_score > start_score + 2:
            return 'growing'
        elif end_score < start_score - 10:
            return 'declining'
        elif end_score < start_score - 2:
            return 'slowing'
        else:
            return 'stable'
    
    def _find_peak_month(self, trajectory: List[Dict]) -> int:
        """Find month when organization reaches peak performance"""
        max_score = max(point['predicted_score'] for point in trajectory)
        for point in trajectory:
            if point['predicted_score'] == max_score:
                return point['month']
        return len(trajectory)
    
    def save_model(self):
        """Save trained model to disk"""
        model_file = self.model_path / "adoption_predictor.joblib"
        scaler_file = self.model_path / "adoption_scaler.joblib"
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self):
        """Load trained model from disk"""
        model_file = self.model_path / "adoption_predictor.joblib"
        scaler_file = self.model_path / "adoption_scaler.joblib"
        
        if model_file.exists() and scaler_file.exists():
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            logger.info("Model loaded successfully")
        else:
            logger.warning("No saved model found. Training required.")

class RiskScorer:
    """Assess risk for organizations falling behind in AI adoption"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def calculate_risk_score(self, org: Dict, peer_data: List[Dict]) -> Dict:
        """Calculate comprehensive risk score for an organization"""
        
        # Current performance vs peers
        peer_scores = [p.get('ai_adoption_score', 0) for p in peer_data if p.get('ai_adoption_score')]
        peer_median = np.median(peer_scores) if peer_scores else 50
        peer_75th = np.percentile(peer_scores, 75) if peer_scores else 60
        
        current_score = org.get('ai_adoption_score', 0)
        
        # Calculate risk factors
        risk_factors = {
            'performance_gap': max(0, (peer_median - current_score) / 100),
            'stagnation_risk': self._assess_stagnation(org),
            'competitive_risk': max(0, (peer_75th - current_score) / 100),
            'resource_risk': self._assess_resource_constraints(org),
            'sector_risk': self._assess_sector_specific_risks(org)
        }
        
        # Weighted risk calculation
        weights = {
            'performance_gap': 0.3,
            'stagnation_risk': 0.25,
            'competitive_risk': 0.2,
            'resource_risk': 0.15,
            'sector_risk': 0.1
        }
        
        overall_risk = sum(risk_factors[factor] * weights[factor] 
                          for factor in risk_factors)
        
        # Determine risk level
        if overall_risk < self.risk_thresholds['low']:
            risk_level = 'low'
        elif overall_risk < self.risk_thresholds['medium']:
            risk_level = 'medium'
        elif overall_risk < self.risk_thresholds['high']:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        return {
            'overall_risk_score': round(overall_risk, 3),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': self._generate_risk_recommendations(risk_level, risk_factors),
            'peer_comparison': {
                'vs_median': current_score - peer_median,
                'vs_75th_percentile': current_score - peer_75th,
                'percentile_rank': self._calculate_percentile_rank(current_score, peer_scores)
            }
        }
    
    def _assess_stagnation(self, org: Dict) -> float:
        """Assess if organization is stagnating in AI adoption"""
        # Look at score velocity and tool adoption rate
        score_change = org.get('score_change_6m', 0)  # 6-month change
        tools_growth = org.get('tools_added_6m', 0)   # Tools added in 6 months
        
        if score_change < 2 and tools_growth == 0:
            return 0.8  # High stagnation risk
        elif score_change < 5:
            return 0.4  # Medium stagnation risk
        else:
            return 0.1  # Low stagnation risk
    
    def _assess_resource_constraints(self, org: Dict) -> float:
        """Assess resource constraint risks"""
        size_band = org.get('size_band', 'medium')
        
        # Smaller organizations typically have higher resource risks
        size_risk_map = {
            'startup': 0.7,
            'small': 0.5,
            'medium': 0.3,
            'large': 0.2,
            'enterprise': 0.1
        }
        
        return size_risk_map.get(size_band, 0.4)
    
    def _assess_sector_specific_risks(self, org: Dict) -> float:
        """Assess sector-specific AI adoption risks"""
        sector = org.get('sector', 'Unknown')
        
        # Different sectors have different AI adoption urgencies
        sector_urgency = {
            'Media': 0.3,           # High AI disruption
            'Communications': 0.2,   # Medium-high AI integration
            'Healthcare': 0.4,       # High regulatory + opportunity
            'Finance': 0.5,          # Critical competitive advantage
            'Retail': 0.3,           # Customer experience critical
            'Manufacturing': 0.2     # Gradual adoption acceptable
        }
        
        return sector_urgency.get(sector, 0.3)
    
    def _generate_risk_recommendations(self, risk_level: str, risk_factors: Dict) -> List[str]:
        """Generate specific recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level in ['high', 'critical']:
            recommendations.append("Immediate action required: Develop emergency AI adoption plan")
            recommendations.append("Consider hiring AI consultants or dedicated AI team")
        
        if risk_factors['performance_gap'] > 0.3:
            recommendations.append("Address performance gap: Benchmark against top-performing peers")
        
        if risk_factors['stagnation_risk'] > 0.5:
            recommendations.append("Break stagnation: Launch new AI pilot projects immediately")
        
        if risk_factors['competitive_risk'] > 0.3:
            recommendations.append("Competitive threat: Accelerate AI implementation timeline")
        
        if risk_factors['resource_risk'] > 0.5:
            recommendations.append("Resource planning: Secure dedicated AI budget and personnel")
        
        return recommendations
    
    def _calculate_percentile_rank(self, score: float, peer_scores: List[float]) -> float:
        """Calculate percentile rank among peers"""
        if not peer_scores:
            return 50.0
        
        return (sum(1 for p in peer_scores if p <= score) / len(peer_scores)) * 100

class ROIEstimator:
    """Estimate ROI for AI adoption initiatives"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.industry_benchmarks = {
            'Media': {
                'automation_savings': 0.15,    # 15% operational cost reduction
                'efficiency_gains': 0.25,      # 25% productivity increase
                'revenue_uplift': 0.08         # 8% revenue increase
            },
            'Communications': {
                'automation_savings': 0.12,
                'efficiency_gains': 0.20,
                'revenue_uplift': 0.10
            }
        }
    
    def estimate_roi(self, org: Dict, investment_scenario: Dict) -> Dict:
        """Estimate ROI for AI adoption investment"""
        
        sector = org.get('sector', 'Media')
        current_score = org.get('ai_adoption_score', 0)
        org_size = org.get('size_band', 'medium')
        
        # Investment parameters
        investment_amount = investment_scenario.get('investment_usd', 100000)
        timeline_months = investment_scenario.get('timeline_months', 12)
        target_score = investment_scenario.get('target_score', current_score + 20)
        
        # Calculate benefits
        benefits = self._calculate_benefits(org, sector, current_score, target_score, org_size)
        
        # Calculate total ROI
        total_benefits = sum(benefits.values())
        roi_percentage = ((total_benefits - investment_amount) / investment_amount) * 100
        
        # Payback period
        monthly_benefits = total_benefits / timeline_months
        payback_months = investment_amount / monthly_benefits if monthly_benefits > 0 else float('inf')
        
        return {
            'investment_amount': investment_amount,
            'timeline_months': timeline_months,
            'benefits_breakdown': benefits,
            'total_benefits': total_benefits,
            'net_benefit': total_benefits - investment_amount,
            'roi_percentage': round(roi_percentage, 1),
            'payback_months': round(payback_months, 1) if payback_months != float('inf') else None,
            'confidence_level': self._calculate_confidence_level(org, investment_scenario),
            'assumptions': self._list_assumptions(sector, org_size)
        }
    
    def _calculate_benefits(self, org: Dict, sector: str, current_score: float, 
                          target_score: float, org_size: str) -> Dict:
        """Calculate detailed benefits breakdown"""
        
        benchmarks = self.industry_benchmarks.get(sector, self.industry_benchmarks['Media'])
        score_improvement = target_score - current_score
        
        # Estimate annual revenue (rough estimates by size)
        revenue_estimates = {
            'startup': 500000,
            'small': 2000000,
            'medium': 10000000,
            'large': 50000000,
            'enterprise': 200000000
        }
        
        annual_revenue = revenue_estimates.get(org_size, 10000000)
        annual_costs = annual_revenue * 0.7  # Assume 70% cost structure
        
        # Scale benefits by score improvement
        improvement_factor = min(score_improvement / 50, 1.0)  # Cap at 50-point improvement
        
        benefits = {
            'automation_savings': annual_costs * benchmarks['automation_savings'] * improvement_factor,
            'efficiency_gains': annual_revenue * benchmarks['efficiency_gains'] * improvement_factor,
            'revenue_uplift': annual_revenue * benchmarks['revenue_uplift'] * improvement_factor,
            'competitive_advantage': annual_revenue * 0.02 * improvement_factor,  # 2% competitive premium
            'risk_mitigation': annual_revenue * 0.01 * improvement_factor         # 1% risk reduction value
        }
        
        return benefits
    
    def _calculate_confidence_level(self, org: Dict, investment_scenario: Dict) -> str:
        """Calculate confidence level in ROI estimate"""
        
        # Factors affecting confidence
        factors = {
            'org_maturity': 0.8 if org.get('ai_adoption_score', 0) > 30 else 0.5,
            'realistic_timeline': 0.8 if investment_scenario.get('timeline_months', 12) >= 6 else 0.4,
            'adequate_investment': 0.8 if investment_scenario.get('investment_usd', 0) > 50000 else 0.5,
            'sector_data': 0.8 if org.get('sector') in self.industry_benchmarks else 0.6
        }
        
        confidence = np.mean(list(factors.values()))
        
        if confidence > 0.75:
            return 'high'
        elif confidence > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _list_assumptions(self, sector: str, org_size: str) -> List[str]:
        """List key assumptions in ROI calculation"""
        return [
            f"Industry benchmarks based on {sector} sector averages",
            f"Revenue estimates based on {org_size} organization typical size",
            "Benefits scale linearly with AI adoption score improvement",
            "Implementation follows industry best practices",
            "Market conditions remain stable during investment period",
            "Organization has necessary change management capabilities"
        ]
