"""
AIMAP ML Data Generator
Generate synthetic training data for ML models based on real patterns
"""
import numpy as np
import pandas as pd
from typing import List, Dict
import random
from datetime import datetime, timedelta
from ..models import Organisation, Metrics, db
from ..scoring.engine import ScoringEngine

class TrainingDataGenerator:
    """Generate synthetic training data for ML models"""
    
    def __init__(self):
        self.scoring_engine = ScoringEngine()
        
        # Define realistic ranges for synthetic data
        self.sector_tools_ranges = {
            'Media': {
                'transcription_tools': (0, 3),
                'genai_copydesk_tools': (0, 5),
                'personalization_signals': (0, 2),
                'training_mentions': (0, 3),
                'policy_documents': (0, 2),
                'total_ai_tools': (0, 8),
                'automation_mentions': (0, 4),
                'governance_mentions': (0, 3)
            },
            'Communications': {
                'press_workflow_ai': (0, 4),
                'content_automation_tools': (0, 5),
                'media_generation_tools': (0, 3),
                'ai_analytics_tools': (0, 3),
                'ai_disclosure_policy': (0, 2),
                'total_ai_tools': (0, 7),
                'training_mentions': (0, 3)
            }
        }
        
        # Maturity progression patterns
        self.maturity_patterns = {
            'Exploring': {'score_range': (0, 25), 'velocity_range': (0, 3)},
            'Piloting': {'score_range': (20, 50), 'velocity_range': (2, 8)},
            'Scaling': {'score_range': (45, 75), 'velocity_range': (3, 12)},
            'Optimizing': {'score_range': (70, 90), 'velocity_range': (2, 6)},
            'Leading': {'score_range': (85, 100), 'velocity_range': (0, 3)}
        }
    
    def generate_historical_data(self, org: Organisation, months_back: int = 12) -> List[Dict]:
        """Generate historical data points for an organization"""
        
        # Get current metrics or create baseline
        current_metrics = Metrics.query.filter_by(organisation_id=org.id).first()
        current_score = current_metrics.ai_adoption_score if current_metrics else random.uniform(10, 60)
        
        # Determine organization's growth trajectory
        trajectory_type = random.choice(['steady_growth', 'rapid_growth', 'stagnant', 'decline_recovery'])
        
        historical_data = []
        
        for month in range(months_back, 0, -1):
            date = datetime.now() - timedelta(days=month * 30)
            period = date.strftime("%Y-%m")
            
            # Calculate historical score based on trajectory
            progress_ratio = (months_back - month) / months_back
            historical_score = self._calculate_historical_score(
                current_score, progress_ratio, trajectory_type
            )
            
            # Generate signals for this historical point
            signals = self._generate_signals_for_score(org.sector, historical_score)
            
            # Calculate additional features
            data_point = {
                'organisation_id': org.id,
                'period': period,
                'ai_adoption_score': historical_score,
                'signals': signals,
                'sector': org.sector,
                'size_band': org.size_band or 'medium',
                'region': org.region or 'Unknown',
                'ai_tools': self._extract_tools_from_signals(signals),
                'months_active': months_back - month + 1,
                'score_velocity': self._calculate_velocity(historical_data, historical_score),
                'peer_ranking': random.uniform(20, 80)  # Will be calculated properly later
            }
            
            historical_data.append(data_point)
        
        return historical_data
    
    def _calculate_historical_score(self, current_score: float, progress_ratio: float, 
                                  trajectory_type: str) -> float:
        """Calculate historical score based on trajectory type"""
        
        if trajectory_type == 'steady_growth':
            # Linear growth with some noise
            historical_score = current_score * (0.3 + 0.7 * progress_ratio)
            historical_score += random.uniform(-5, 5)
            
        elif trajectory_type == 'rapid_growth':
            # Exponential growth curve
            historical_score = current_score * (0.2 + 0.8 * (progress_ratio ** 0.5))
            historical_score += random.uniform(-3, 3)
            
        elif trajectory_type == 'stagnant':
            # Mostly flat with minor fluctuations
            base_score = current_score * 0.8
            historical_score = base_score + random.uniform(-10, 10)
            
        else:  # decline_recovery
            # Initial decline then recovery
            if progress_ratio < 0.6:
                # Decline phase
                historical_score = current_score * (0.4 + 0.4 * progress_ratio)
            else:
                # Recovery phase
                recovery_ratio = (progress_ratio - 0.6) / 0.4
                historical_score = current_score * (0.6 + 0.4 * recovery_ratio)
            
            historical_score += random.uniform(-8, 8)
        
        return max(0, min(100, historical_score))
    
    def _generate_signals_for_score(self, sector: str, target_score: float) -> Dict:
        """Generate realistic signals that would produce the target score"""
        
        if sector not in self.sector_tools_ranges:
            sector = 'Media'  # Default fallback
        
        ranges = self.sector_tools_ranges[sector]
        signals = {}
        
        # Scale signal strength based on target score
        score_factor = target_score / 100.0
        
        for signal, (min_val, max_val) in ranges.items():
            # Higher scores should have higher signal values
            if score_factor < 0.2:
                # Low scores - mostly zeros and low values
                signals[signal] = random.choice([0, 0, 0, random.randint(min_val, min_val + 1)])
            elif score_factor < 0.5:
                # Medium-low scores
                signals[signal] = random.randint(min_val, int((max_val - min_val) * 0.4) + min_val)
            elif score_factor < 0.8:
                # Medium-high scores
                signals[signal] = random.randint(int((max_val - min_val) * 0.3) + min_val, 
                                               int((max_val - min_val) * 0.8) + min_val)
            else:
                # High scores - higher values
                signals[signal] = random.randint(int((max_val - min_val) * 0.6) + min_val, max_val)
        
        return signals
    
    def _extract_tools_from_signals(self, signals: Dict) -> List[str]:
        """Extract AI tools list based on signals"""
        total_tools = signals.get('total_ai_tools', 0)
        
        possible_tools = [
            'chatgpt', 'gemini', 'claude', 'midjourney', 'dalle',
            'grammarly', 'otter.ai', 'elevenlabs', 'canva-ai',
            'notion-ai', 'jasper', 'copy.ai'
        ]
        
        if total_tools == 0:
            return []
        
        # Select random tools up to the total count
        selected_tools = random.sample(possible_tools, min(total_tools, len(possible_tools)))
        return selected_tools
    
    def _calculate_velocity(self, historical_data: List[Dict], current_score: float) -> float:
        """Calculate score velocity (change rate)"""
        if len(historical_data) < 2:
            return 0.0
        
        # Compare with 3 months ago
        if len(historical_data) >= 3:
            old_score = historical_data[-3]['ai_adoption_score']
            velocity = (current_score - old_score) / 3.0  # Points per month
        else:
            old_score = historical_data[0]['ai_adoption_score']
            velocity = (current_score - old_score) / len(historical_data)
        
        return velocity
    
    def generate_peer_data(self, org: Organisation, num_peers: int = 20) -> List[Dict]:
        """Generate peer organization data for benchmarking"""
        peers = []
        
        for i in range(num_peers):
            # Create peer with similar characteristics but varied performance
            peer_score = random.uniform(10, 95)
            
            peer = {
                'id': f"peer_{org.id}_{i}",
                'sector': org.sector,
                'size_band': random.choice(['startup', 'small', 'medium', 'large', 'enterprise']),
                'region': random.choice(['North America', 'Europe', 'Asia Pacific', 'Latin America']),
                'ai_adoption_score': peer_score,
                'ai_tools': self._extract_tools_from_signals(
                    self._generate_signals_for_score(org.sector, peer_score)
                ),
                'maturity_stage': self._score_to_maturity(peer_score)
            }
            
            peers.append(peer)
        
        return peers
    
    def _score_to_maturity(self, score: float) -> str:
        """Convert score to maturity stage"""
        if score < 20:
            return 'Exploring'
        elif score < 45:
            return 'Piloting'
        elif score < 70:
            return 'Scaling'
        elif score < 85:
            return 'Optimizing'
        else:
            return 'Leading'
    
    def create_training_dataset(self, min_orgs: int = 50) -> pd.DataFrame:
        """Create a comprehensive training dataset"""
        
        # Get existing organizations
        existing_orgs = Organisation.query.all()
        
        # Generate additional synthetic organizations if needed
        all_training_data = []
        
        for org in existing_orgs:
            # Generate historical data for each org
            historical_data = self.generate_historical_data(org, months_back=18)
            all_training_data.extend(historical_data)
        
        # Generate additional synthetic organizations if we don't have enough data
        if len(all_training_data) < min_orgs * 10:  # Want ~10 data points per org
            additional_orgs_needed = (min_orgs * 10 - len(all_training_data)) // 10
            synthetic_data = self._generate_synthetic_organizations(additional_orgs_needed)
            all_training_data.extend(synthetic_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_training_data)
        
        # Add derived features
        df['score_change_3m'] = df.groupby('organisation_id')['ai_adoption_score'].diff(3).fillna(0)
        df['tools_added_6m'] = df.groupby('organisation_id')['ai_tools'].apply(
            lambda x: [len(tools) for tools in x]
        ).apply(lambda x: pd.Series(x).diff(6).fillna(0))
        
        return df
    
    def _generate_synthetic_organizations(self, count: int) -> List[Dict]:
        """Generate completely synthetic organizations and their data"""
        synthetic_data = []
        
        for org_id in range(1000, 1000 + count):  # Use high IDs to avoid conflicts
            # Create synthetic org profile
            sector = random.choice(['Media', 'Communications'])
            size_band = random.choice(['startup', 'small', 'medium', 'large', 'enterprise'])
            region = random.choice(['North America', 'Europe', 'Asia Pacific', 'Latin America'])
            
            # Generate 12-18 months of data for this synthetic org
            months_data = random.randint(12, 18)
            org_trajectory = random.choice(['steady_growth', 'rapid_growth', 'stagnant', 'decline_recovery'])
            current_score = random.uniform(15, 90)
            
            for month in range(months_data):
                date = datetime.now() - timedelta(days=(months_data - month) * 30)
                period = date.strftime("%Y-%m")
                
                progress_ratio = month / months_data
                historical_score = self._calculate_historical_score(
                    current_score, progress_ratio, org_trajectory
                )
                
                signals = self._generate_signals_for_score(sector, historical_score)
                
                data_point = {
                    'organisation_id': org_id,
                    'period': period,
                    'ai_adoption_score': historical_score,
                    'signals': signals,
                    'sector': sector,
                    'size_band': size_band,
                    'region': region,
                    'ai_tools': self._extract_tools_from_signals(signals),
                    'months_active': month + 1,
                    'score_velocity': random.uniform(-2, 5),
                    'peer_ranking': random.uniform(20, 80)
                }
                
                synthetic_data.append(data_point)
        
        return synthetic_data
