"""
AIMAP Scoring Engine
Core scoring and benchmarking functionality
"""
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
from ..models import Organisation, Metrics, db

class ScoringEngine:
    """Main scoring engine for AI adoption metrics"""
    
    def __init__(self):
        self.benchmarks = self._load_benchmarks()
        self.sector_adapters = {
            'Media': MediaAdapter(),
            'Communications': CommunicationsAdapter()
        }
    
    def _load_benchmarks(self) -> Dict:
        """Load benchmarks configuration from YAML"""
        config_path = Path(__file__).parent / "benchmarks.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def score_organisation(self, org: Organisation, signals: Dict, period: str) -> Tuple[float, str]:
        """
        Score an organisation based on signals
        Returns: (score, maturity_stage)
        """
        if org.sector not in self.sector_adapters:
            raise ValueError(f"No adapter for sector: {org.sector}")
        
        adapter = self.sector_adapters[org.sector]
        features = adapter.extract_features(signals)
        
        sector_config = self.benchmarks['sectors'][org.sector]
        weights = sector_config['weights']
        
        # Calculate weighted score
        weighted_sum = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in features:
                weighted_sum += features[feature] * weight
                total_weight += weight
        
        # Normalize to 0-100 scale
        if total_weight > 0:
            score = (weighted_sum / total_weight) * 100
        else:
            score = 0.0
        
        # Determine maturity stage
        thresholds = sector_config['maturity_thresholds']
        stages = ['Exploring', 'Piloting', 'Scaling', 'Optimizing', 'Leading']
        
        stage = stages[0]  # Default to Exploring
        for i, threshold in enumerate(thresholds):
            if score >= threshold:
                stage = stages[i + 1]
        
        return round(score, 2), stage
    
    def create_benchmark_bucket(self, org: Organisation) -> str:
        """Create benchmark bucket identifier"""
        return f"{org.sector}:{org.region or 'Unknown'}:{org.size_band or 'Unknown'}"
    
    def get_peer_benchmarks(self, bucket: str, period: str) -> Dict:
        """Get peer benchmark statistics for a bucket"""
        metrics = Metrics.query.filter_by(
            benchmark_bucket=bucket,
            period=period
        ).filter(
            Metrics.ai_adoption_score.isnot(None)
        ).all()
        
        if not metrics:
            return {
                'bucket': bucket,
                'median_score': 0.0,
                'p25_score': 0.0,
                'p75_score': 0.0,
                'count': 0
            }
        
        scores = [m.ai_adoption_score for m in metrics]
        
        return {
            'bucket': bucket,
            'median_score': statistics.median(scores),
            'p25_score': statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else min(scores),
            'p75_score': statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else max(scores),
            'count': len(scores)
        }
    
    def get_recommendations(self, org: Organisation, features: Dict, gaps: List[str]) -> List[str]:
        """Get improvement recommendations for an organisation"""
        if org.sector not in self.sector_adapters:
            return ["Contact AIMAP team for sector-specific recommendations"]
        
        adapter = self.sector_adapters[org.sector]
        return adapter.recommendations(features, gaps)
    
    def identify_gaps(self, features: Dict, sector: str) -> List[str]:
        """Identify areas for improvement based on low feature scores"""
        gaps = []
        sector_config = self.benchmarks['sectors'].get(sector, {})
        weights = sector_config.get('weights', {})
        
        for feature, weight in weights.items():
            if feature in features and features[feature] < 0.5:  # Below 50% threshold
                gaps.append(feature)
        
        return gaps

class SectorAdapter:
    """Base class for sector-specific adapters"""
    
    def extract_features(self, signals: Dict) -> Dict[str, float]:
        """Extract normalized features from raw signals"""
        raise NotImplementedError
    
    def recommendations(self, features: Dict, gaps: List[str]) -> List[str]:
        """Generate recommendations based on features and gaps"""
        raise NotImplementedError

class MediaAdapter(SectorAdapter):
    """Media sector adapter"""
    
    def extract_features(self, signals: Dict) -> Dict[str, float]:
        """Extract media-specific features"""
        features = {}
        
        # Transcription capability
        features['has_transcription'] = float(signals.get('transcription_tools', 0) > 0)
        
        # GenAI in content workflow
        genai_tools = signals.get('genai_copydesk_tools', 0)
        features['genai_in_copydesk'] = min(1.0, genai_tools / 3.0)  # Normalize to 0-1
        
        # Audience personalization
        features['audience_personalization'] = float(signals.get('personalization_signals', 0) > 0)
        
        # Training programs
        training_mentions = signals.get('training_mentions', 0)
        features['newsroom_training'] = min(1.0, training_mentions / 2.0)
        
        # AI policy
        features['ai_policy'] = float(signals.get('policy_documents', 0) > 0)
        
        # Tool footprint
        total_tools = signals.get('total_ai_tools', 0)
        features['tool_footprint'] = min(1.0, total_tools / 5.0)  # Normalize to max 5 tools
        
        # Automation in operations
        automation_score = signals.get('automation_mentions', 0)
        features['automation_in_ops'] = min(1.0, automation_score / 3.0)
        
        # Governance signals
        governance_score = signals.get('governance_mentions', 0)
        features['governance_signals'] = min(1.0, governance_score / 2.0)
        
        return features
    
    def recommendations(self, features: Dict, gaps: List[str]) -> List[str]:
        """Media-specific recommendations"""
        recommendations = []
        
        if 'has_transcription' in gaps:
            recommendations.append("Implement AI transcription tools for audio/video content")
        
        if 'genai_in_copydesk' in gaps:
            recommendations.append("Integrate GenAI tools into editorial workflow")
        
        if 'newsroom_training' in gaps:
            recommendations.append("Develop comprehensive AI training programs for newsroom staff")
        
        if 'ai_policy' in gaps:
            recommendations.append("Establish clear AI governance and ethics policies")
        
        if 'tool_footprint' in gaps:
            recommendations.append("Expand AI tool adoption across different newsroom functions")
        
        if 'automation_in_ops' in gaps:
            recommendations.append("Automate routine operational tasks with AI")
        
        return recommendations[:6]  # Limit to top 6

class CommunicationsAdapter(SectorAdapter):
    """Communications/PR sector adapter"""
    
    def extract_features(self, signals: Dict) -> Dict[str, float]:
        """Extract communications-specific features"""
        features = {}
        
        # AI in press workflow
        press_ai_tools = signals.get('press_workflow_ai', 0)
        features['ai_in_press_workflow'] = min(1.0, press_ai_tools / 3.0)
        
        # Content automation
        content_automation = signals.get('content_automation_tools', 0)
        features['content_automation'] = min(1.0, content_automation / 3.0)
        
        # Image/voice generation
        features['image_voice_gen_use'] = float(signals.get('media_generation_tools', 0) > 0)
        
        # Analytics insights
        analytics_tools = signals.get('ai_analytics_tools', 0)
        features['analytics_insights'] = min(1.0, analytics_tools / 2.0)
        
        # Governance disclosure
        features['governance_disclosure'] = float(signals.get('ai_disclosure_policy', 0) > 0)
        
        # Tool footprint
        total_tools = signals.get('total_ai_tools', 0)
        features['tool_footprint'] = min(1.0, total_tools / 4.0)
        
        # Team training
        training_mentions = signals.get('training_mentions', 0)
        features['team_training'] = min(1.0, training_mentions / 2.0)
        
        return features
    
    def recommendations(self, features: Dict, gaps: List[str]) -> List[str]:
        """Communications-specific recommendations"""
        recommendations = []
        
        if 'ai_in_press_workflow' in gaps:
            recommendations.append("Integrate AI tools into press release and content workflow")
        
        if 'content_automation' in gaps:
            recommendations.append("Implement content automation for social media and campaigns")
        
        if 'image_voice_gen_use' in gaps:
            recommendations.append("Adopt AI-powered image and voice generation tools")
        
        if 'analytics_insights' in gaps:
            recommendations.append("Use AI analytics for campaign performance and audience insights")
        
        if 'governance_disclosure' in gaps:
            recommendations.append("Develop AI disclosure policies for client transparency")
        
        if 'tool_footprint' in gaps:
            recommendations.append("Expand AI tool adoption across PR and communications functions")
        
        if 'team_training' in gaps:
            recommendations.append("Train team on AI tools and best practices")
        
        return recommendations[:6]  # Limit to top 6
