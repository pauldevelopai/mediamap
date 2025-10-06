"""
Test AIMAP Scoring Engine
"""
import pytest
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from aimap.scoring.engine import ScoringEngine
from aimap.models import Organisation, Metrics
from models import db

class TestScoringEngine:
    
    def setup_method(self):
        """Set up test data"""
        self.engine = ScoringEngine()
        
        # Create test organisation
        self.test_org = type('MockOrg', (), {
            'id': 1,
            'name': 'Test Media Corp',
            'sector': 'Media',
            'subsector': 'Digital',
            'region': 'North America',
            'country': 'United States',
            'size_band': 'medium',
            'client_tag': 'tier1'
        })()
    
    def test_load_benchmarks(self):
        """Test benchmark configuration loading"""
        assert 'Media' in self.engine.benchmarks['sectors']
        assert 'Communications' in self.engine.benchmarks['sectors']
        
        media_config = self.engine.benchmarks['sectors']['Media']
        assert 'weights' in media_config
        assert 'maturity_thresholds' in media_config
        assert len(media_config['maturity_thresholds']) == 4
    
    def test_media_scoring(self):
        """Test media sector scoring"""
        test_signals = {
            'transcription_tools': 1,
            'genai_copydesk_tools': 2,
            'personalization_signals': 1,
            'training_mentions': 1,
            'policy_documents': 1,
            'total_ai_tools': 5,
            'automation_mentions': 2,
            'governance_mentions': 1
        }
        
        score, maturity_stage = self.engine.score_organisation(
            self.test_org, test_signals, "2025-08"
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert maturity_stage in ['Exploring', 'Piloting', 'Scaling', 'Optimizing', 'Leading']
    
    def test_communications_scoring(self):
        """Test communications sector scoring"""
        comm_org = type('MockOrg', (), {
            'id': 2,
            'name': 'Test PR Agency',
            'sector': 'Communications',
            'region': 'Europe',
            'size_band': 'small'
        })()
        
        test_signals = {
            'press_workflow_ai': 2,
            'content_automation_tools': 3,
            'media_generation_tools': 1,
            'ai_analytics_tools': 1,
            'ai_disclosure_policy': 1,
            'total_ai_tools': 4,
            'training_mentions': 1
        }
        
        score, maturity_stage = self.engine.score_organisation(
            comm_org, test_signals, "2025-08"
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert maturity_stage in ['Exploring', 'Piloting', 'Scaling', 'Optimizing', 'Leading']
    
    def test_benchmark_bucket_creation(self):
        """Test benchmark bucket creation"""
        bucket = self.engine.create_benchmark_bucket(self.test_org)
        expected = "Media:North America:medium"
        assert bucket == expected
    
    def test_deterministic_scoring(self):
        """Test that scoring is deterministic"""
        test_signals = {
            'transcription_tools': 1,
            'genai_copydesk_tools': 2,
            'total_ai_tools': 3
        }
        
        score1, stage1 = self.engine.score_organisation(
            self.test_org, test_signals, "2025-08"
        )
        score2, stage2 = self.engine.score_organisation(
            self.test_org, test_signals, "2025-08"
        )
        
        assert score1 == score2
        assert stage1 == stage2
    
    def test_empty_signals(self):
        """Test scoring with empty signals"""
        score, maturity_stage = self.engine.score_organisation(
            self.test_org, {}, "2025-08"
        )
        
        assert score == 0.0
        assert maturity_stage == 'Exploring'
    
    def test_gap_identification(self):
        """Test gap identification"""
        features = {
            'transcription_tools': 0.2,  # Below 0.5 threshold
            'genai_copydesk_tools': 0.8,  # Above 0.5 threshold
            'total_ai_tools': 0.3  # Below 0.5 threshold
        }
        
        gaps = self.engine.identify_gaps(features, 'Media')
        
        # Should identify features below 0.5 as gaps
        expected_gaps = ['transcription_tools', 'total_ai_tools']
        for gap in expected_gaps:
            if gap in features:  # Only check if feature is in our test data
                assert gap in gaps
    
    def test_sector_adapters(self):
        """Test sector adapter availability"""
        assert 'Media' in self.engine.sector_adapters
        assert 'Communications' in self.engine.sector_adapters
        
        media_adapter = self.engine.sector_adapters['Media']
        assert hasattr(media_adapter, 'extract_features')
        assert hasattr(media_adapter, 'recommendations')

if __name__ == '__main__':
    pytest.main([__file__])
