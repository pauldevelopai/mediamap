"""
Media Sector Adapter
Feature extraction and recommendations for media organizations
"""
from typing import Dict, List

def extract_features(signals: Dict) -> Dict[str, float]:
    """Extract media-specific features from signals"""
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

def recommendations(features: Dict, gaps: List[str]) -> List[str]:
    """Generate media-specific recommendations"""
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
    
    if 'audience_personalization' in gaps:
        recommendations.append("Implement AI-driven audience personalization systems")
    
    if 'governance_signals' in gaps:
        recommendations.append("Strengthen AI governance framework and transparency")
    
    return recommendations[:6]  # Limit to top 6
