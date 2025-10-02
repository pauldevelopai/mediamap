"""
Test AIMAP Ingestion Pipeline
"""
import pytest
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from aimap.ingest.transformers import normalize_signals, deduplicate_tools, merge_signals
from aimap.ingest.sources import MediaScraper, CommunicationsScraper

class TestDataTransformers:
    
    def test_normalize_signals(self):
        """Test signal normalization"""
        raw_signals = {
            'transcription_tools': 2,
            'total_ai_tools': -1,  # Should be clamped to 0
            'detected_tools': ['ChatGPT', 'GPT-4', 'ChatGPT'],  # Should be deduplicated
            'text_field': '  whitespace  ',  # Should be stripped
            'empty_list': []
        }
        
        normalized = normalize_signals(raw_signals)
        
        assert normalized['transcription_tools'] == 2
        assert normalized['total_ai_tools'] == 0  # Negative clamped
        assert len(normalized['detected_tools']) == 2  # Deduplicated
        assert normalized['text_field'] == 'whitespace'  # Stripped
        assert normalized['empty_list'] == []
    
    def test_deduplicate_tools(self):
        """Test AI tool deduplication"""
        tools = ['ChatGPT', 'GPT-4', 'chatgpt', 'OpenAI', 'Gemini', 'GPT-3']
        
        deduplicated = deduplicate_tools(tools)
        
        # Should normalize and deduplicate
        assert 'chatgpt' in deduplicated
        assert 'gemini' in deduplicated
        
        # Should not have duplicates
        assert len(deduplicated) == len(set(deduplicated))
        
        # Should be sorted
        assert deduplicated == sorted(deduplicated)
    
    def test_merge_signals(self):
        """Test signal merging"""
        old_signals = {
            'transcription_tools': 1,
            'detected_tools': ['ChatGPT'],
            'boolean_field': False
        }
        
        new_signals = {
            'transcription_tools': 2,  # Should take max
            'detected_tools': ['Gemini', 'ChatGPT'],  # Should combine
            'boolean_field': True,  # Should OR
            'new_field': 5
        }
        
        merged = merge_signals(old_signals, new_signals)
        
        assert merged['transcription_tools'] == 2  # Max
        assert 'chatgpt' in merged['detected_tools']  # Combined
        assert 'gemini' in merged['detected_tools']
        assert merged['boolean_field'] is True  # OR logic
        assert merged['new_field'] == 5  # New field added

class TestWebScrapers:
    
    def test_media_scraper_initialization(self):
        """Test media scraper initialization"""
        scraper = MediaScraper()
        assert hasattr(scraper, 'session')
        assert hasattr(scraper, 'detect_ai_tools')
    
    def test_communications_scraper_initialization(self):
        """Test communications scraper initialization"""
        scraper = CommunicationsScraper()
        assert hasattr(scraper, 'session')
        assert hasattr(scraper, 'discover_press_page')
        assert hasattr(scraper, 'discover_jobs_page')
    
    def test_ai_tool_detection(self):
        """Test AI tool detection in text"""
        scraper = MediaScraper()
        
        text = """
        We use ChatGPT for content generation and Midjourney for image creation.
        Our team also leverages GPT-4 for research and Claude for analysis.
        """
        
        tools = scraper.detect_ai_tools(text)
        
        expected_tools = ['chatgpt', 'midjourney', 'claude']
        for tool in expected_tools:
            assert any(tool in detected.lower() for detected in tools)
    
    def test_media_signal_detection(self):
        """Test media-specific signal detection"""
        scraper = MediaScraper()
        
        # Test transcription signals
        text = "transcription speech-to-text subtitle caption"
        count = scraper._count_transcription_signals(text)
        assert count == 4
        
        # Test automation signals
        text = "automation automated workflow automation process automation"
        count = scraper._count_automation_signals(text)
        assert count == 4
    
    def test_communications_signal_detection(self):
        """Test communications-specific signal detection"""
        scraper = CommunicationsScraper()
        
        # Test press workflow signals
        text = "ai press automated press ai content creation ai copywriting"
        count = scraper._count_press_workflow_signals(text)
        assert count == 4
        
        # Test content automation signals
        text = "content automation social media automation campaign automation"
        count = scraper._count_content_automation_signals(text)
        assert count == 3

class TestIngestionPipeline:
    
    def test_pipeline_initialization(self):
        """Test ingestion pipeline initialization"""
        # This would require a Flask app context and database setup
        # For now, just test that imports work
        from aimap.ingest.pipeline import IngestionPipeline
        assert IngestionPipeline is not None

if __name__ == '__main__':
    pytest.main([__file__])
