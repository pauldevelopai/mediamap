"""
Communications/PR Sector Adapter
Feature extraction and recommendations for communications organizations
"""
from typing import Dict, List

def extract_features(signals: Dict) -> Dict[str, float]:
    """Extract communications-specific features from signals"""
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

def recommendations(features: Dict, gaps: List[str]) -> List[str]:
    """Generate communications-specific recommendations"""
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
