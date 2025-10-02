"""
Test AIMAP Reports
"""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

class TestPPTXReportGenerator:
    
    def test_maturity_stage_roadmap(self):
        """Test roadmap generation for different maturity stages"""
        # Import here to avoid Flask app context issues in tests
        try:
            from aimap.reports.pptx_export import PPTXReportGenerator
            
            generator = PPTXReportGenerator()
            
            # Test different stages
            stages = ['Exploring', 'Piloting', 'Scaling', 'Optimizing', 'Leading']
            
            for stage in stages:
                roadmap = generator._get_roadmap_for_stage(stage, 'Media')
                assert isinstance(roadmap, list)
                assert len(roadmap) > 0
                
                # Each roadmap item should be a string
                for item in roadmap:
                    assert isinstance(item, str)
                    assert len(item) > 10  # Should be descriptive
            
            # Test unknown stage defaults to Exploring
            unknown_roadmap = generator._get_roadmap_for_stage('Unknown', 'Media')
            exploring_roadmap = generator._get_roadmap_for_stage('Exploring', 'Media')
            assert unknown_roadmap == exploring_roadmap
            
        except ImportError:
            # Skip test if dependencies not available
            pytest.skip("PPTX dependencies not available")

class TestPDFReportGenerator:
    
    def test_css_generation(self):
        """Test CSS generation for PDF reports"""
        try:
            from aimap.reports.pdf_export import PDFReportGenerator
            
            generator = PDFReportGenerator()
            css = generator._get_report_css()
            
            assert isinstance(css, str)
            assert len(css) > 100  # Should be substantial
            
            # Check for key CSS elements
            assert '@page' in css
            assert 'font-family' in css
            assert '.header' in css
            assert '.scorecard' in css
            
        except ImportError:
            # Skip test if dependencies not available
            pytest.skip("PDF dependencies not available")

class TestReportTemplate:
    
    def test_template_exists(self):
        """Test that report template exists"""
        template_path = Path(__file__).parent.parent / "aimap" / "reports" / "templates" / "organisation_report.html"
        assert template_path.exists(), "Report template should exist"
        
        with open(template_path, 'r') as f:
            content = f.read()
            
        # Check for key template elements
        assert '{{' in content  # Jinja2 template syntax
        assert 'org.name' in content
        assert 'metrics.ai_adoption_score' in content
        assert 'benchmark_data' in content

class TestReportData:
    
    def test_mock_organisation_data(self):
        """Test with mock organisation data"""
        # Create mock organisation
        mock_org = Mock()
        mock_org.id = 1
        mock_org.name = "Test Media Corp"
        mock_org.sector = "Media"
        mock_org.region = "North America"
        mock_org.size_band = "medium"
        mock_org.ai_tools = ["chatgpt", "midjourney"]
        mock_org.website_url = "https://testmedia.com"
        
        # Create mock metrics
        mock_metrics = Mock()
        mock_metrics.ai_adoption_score = 67.5
        mock_metrics.maturity_stage = "Scaling"
        mock_metrics.signals = {
            'transcription_tools': 1,
            'genai_copydesk_tools': 2,
            'total_ai_tools': 5
        }
        mock_metrics.benchmark_bucket = "Media:North America:medium"
        mock_metrics.period = "2025-08"
        
        # Test data consistency
        assert mock_org.name
        assert mock_metrics.ai_adoption_score > 0
        assert mock_metrics.maturity_stage in ['Exploring', 'Piloting', 'Scaling', 'Optimizing', 'Leading']
        assert isinstance(mock_metrics.signals, dict)

if __name__ == '__main__':
    pytest.main([__file__])
