"""
HealthPIN Data Coordinator
Centralizes and coordinates all agent data for useful presentation
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class HealthPINDataCoordinator:
    """Coordinates and processes HealthPIN agent data for dashboard display"""
    
    def __init__(self):
        self.agent_data_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_data.json'
        self.insights_file = '/opt/mediamap/backend/agents/storage/healthpin/HealthPINAgent_insights.json'
    
    def load_agent_data(self) -> List[Dict]:
        """Load raw agent data from JSON file"""
        try:
            if os.path.exists(self.agent_data_file):
                with open(self.agent_data_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading agent data: {e}")
            return []
    
    def load_insights_data(self) -> List[Dict]:
        """Load processed insights from JSON file"""
        try:
            if os.path.exists(self.insights_file):
                with open(self.insights_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading insights data: {e}")
            return []
    
    def get_coordinated_dashboard_stats(self) -> Dict[str, Any]:
        """Get coordinated statistics for HealthPIN dashboard"""
        agent_data = self.load_agent_data()
        insights_data = self.load_insights_data()
        
        if not agent_data:
            return self._get_empty_stats()
        
        # Process and categorize the data
        categories = {}
        sources = set()
        recent_entries = []
        clinical_data = []
        research_data = []
        policy_data = []
        
        for entry in agent_data:
            category = entry.get('category', 'Unknown')
            source = entry.get('source', 'Unknown')
            timestamp = entry.get('timestamp', '')
            
            categories[category] = categories.get(category, 0) + 1
            sources.add(source)
            
            # Categorize for useful presentation
            if 'Clinical' in category:
                clinical_data.append(entry)
            elif 'Research' in category:
                research_data.append(entry)
            elif 'Policy' in category:
                policy_data.append(entry)
            
            # Keep recent entries
            if len(recent_entries) < 10:
                recent_entries.append(entry)
        
        # Create coordinated statistics
        stats = {
            # Main dashboard metrics (mapped to existing UI)
            'total_patients': len(clinical_data),  # Clinical care entries as "patients"
            'total_doctors': len(sources),  # Data sources as "doctors"
            'total_records': len(agent_data),  # Total healthcare entries
            'total_matches': len(categories),  # Categories as "AI matches"
            
            # Detailed breakdowns
            'categories': categories,
            'sources': list(sources),
            'recent_entries': recent_entries[-5:],  # Last 5 entries
            
            # Coordinated data for useful presentation
            'clinical_insights': self._process_clinical_data(clinical_data),
            'research_findings': self._process_research_data(research_data),
            'policy_updates': self._process_policy_data(policy_data),
            
            # AI insights
            'ai_insights': insights_data[-10:] if insights_data else [],
            
            # System status
            'data_freshness': self._get_data_freshness(agent_data),
            'collection_status': 'active' if agent_data else 'inactive',
            'last_update': agent_data[-1].get('timestamp', '') if agent_data else '',
            
            # Useful metrics
            'growth_trend': self._calculate_growth_trend(agent_data),
            'top_categories': self._get_top_categories(categories),
            'data_quality_score': self._calculate_data_quality(agent_data)
        }
        
        return stats
    
    def _get_empty_stats(self) -> Dict[str, Any]:
        """Return empty stats structure"""
        return {
            'total_patients': 0,
            'total_doctors': 0,
            'total_records': 0,
            'total_matches': 0,
            'categories': {},
            'sources': [],
            'recent_entries': [],
            'clinical_insights': [],
            'research_findings': [],
            'policy_updates': [],
            'ai_insights': [],
            'data_freshness': 'no_data',
            'collection_status': 'inactive',
            'last_update': '',
            'growth_trend': 'stable',
            'top_categories': [],
            'data_quality_score': 0
        }
    
    def _process_clinical_data(self, clinical_data: List[Dict]) -> List[Dict]:
        """Process clinical care data for useful presentation"""
        processed = []
        for entry in clinical_data[-5:]:  # Last 5 clinical entries
            processed.append({
                'title': f"Clinical Care Update",
                'description': entry.get('insight', 'Healthcare data collected')[:100] + '...',
                'source': entry.get('source', 'Unknown'),
                'timestamp': entry.get('timestamp', ''),
                'category': entry.get('category', 'Clinical_Care'),
                'confidence': entry.get('confidence', 0.8)
            })
        return processed
    
    def _process_research_data(self, research_data: List[Dict]) -> List[Dict]:
        """Process medical research data for useful presentation"""
        processed = []
        for entry in research_data[-5:]:  # Last 5 research entries
            processed.append({
                'title': f"Medical Research Finding",
                'description': entry.get('insight', 'Research data collected')[:100] + '...',
                'source': entry.get('source', 'Unknown'),
                'timestamp': entry.get('timestamp', ''),
                'category': entry.get('category', 'Medical_Research'),
                'confidence': entry.get('confidence', 0.8)
            })
        return processed
    
    def _process_policy_data(self, policy_data: List[Dict]) -> List[Dict]:
        """Process healthcare policy data for useful presentation"""
        processed = []
        for entry in policy_data[-5:]:  # Last 5 policy entries
            processed.append({
                'title': f"Healthcare Policy Update",
                'description': entry.get('insight', 'Policy data collected')[:100] + '...',
                'source': entry.get('source', 'Unknown'),
                'timestamp': entry.get('timestamp', ''),
                'category': entry.get('category', 'Healthcare_Policy'),
                'confidence': entry.get('confidence', 0.8)
            })
        return processed
    
    def _get_data_freshness(self, agent_data: List[Dict]) -> str:
        """Determine how fresh the data is"""
        if not agent_data:
            return 'no_data'
        
        try:
            last_entry = agent_data[-1]
            last_timestamp = last_entry.get('timestamp', '')
            if last_timestamp:
                last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                now = datetime.now(last_time.tzinfo)
                hours_diff = (now - last_time).total_seconds() / 3600
                
                if hours_diff < 1:
                    return 'very_fresh'
                elif hours_diff < 24:
                    return 'fresh'
                elif hours_diff < 72:
                    return 'recent'
                else:
                    return 'stale'
        except:
            pass
        
        return 'unknown'
    
    def _calculate_growth_trend(self, agent_data: List[Dict]) -> str:
        """Calculate data collection growth trend"""
        if len(agent_data) < 10:
            return 'insufficient_data'
        
        # Simple trend calculation based on recent vs older data
        recent_count = len([e for e in agent_data[-20:]])  # Last 20 entries
        older_count = len([e for e in agent_data[-40:-20]])  # Previous 20 entries
        
        if recent_count > older_count * 1.2:
            return 'growing'
        elif recent_count < older_count * 0.8:
            return 'declining'
        else:
            return 'stable'
    
    def _get_top_categories(self, categories: Dict[str, int]) -> List[Dict]:
        """Get top categories by count"""
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        return [{'name': cat, 'count': count} for cat, count in sorted_categories[:5]]
    
    def _calculate_data_quality(self, agent_data: List[Dict]) -> float:
        """Calculate a data quality score (0-100)"""
        if not agent_data:
            return 0
        
        quality_factors = []
        
        # Check for required fields
        required_fields = ['category', 'source', 'timestamp']
        for entry in agent_data[-10:]:  # Check last 10 entries
            field_score = sum(1 for field in required_fields if entry.get(field))
            quality_factors.append(field_score / len(required_fields))
        
        # Check for diversity in sources and categories
        sources = set(entry.get('source') for entry in agent_data)
        categories = set(entry.get('category') for entry in agent_data)
        
        diversity_score = min(len(sources) / 5, 1) * 0.3 + min(len(categories) / 4, 1) * 0.3
        completeness_score = sum(quality_factors) / len(quality_factors) * 0.4
        
        return round((diversity_score + completeness_score) * 100, 1)

# Global instance
healthpin_coordinator = HealthPINDataCoordinator()
