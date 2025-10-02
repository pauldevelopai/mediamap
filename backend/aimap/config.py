"""
AIMAP Configuration
Central configuration management for AIMAP
"""
import os
from pathlib import Path

# Database
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///aimap.db')

# OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MEDIA_ROOT = PROJECT_ROOT / "_aimap_media"
REPORTS_ROOT = PROJECT_ROOT / "_aimap_reports"

# Ensure directories exist
MEDIA_ROOT.mkdir(exist_ok=True)
REPORTS_ROOT.mkdir(exist_ok=True)

# Scoring
DEFAULT_PERIOD = "2025-08"

# Export settings
PPTX_TEMPLATE_PATH = PROJECT_ROOT / "backend" / "aimap" / "reports" / "templates"
PDF_TEMPLATE_PATH = PROJECT_ROOT / "backend" / "aimap" / "reports" / "templates"

# Ingestion settings
USER_AGENT = "AIMAP/1.0 (+https://aimap.ai)"
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1  # seconds between requests

# AI Tools detection patterns
AI_TOOLS_PATTERNS = [
    'chatgpt', 'gpt-4', 'gpt-3', 'openai',
    'gemini', 'bard', 'claude', 'anthropic',
    'midjourney', 'dall-e', 'stable diffusion',
    'elevenlabs', 'murf', 'synthesia',
    'jasper', 'copy.ai', 'writesonic',
    'grammarly', 'notion ai', 'clickup ai',
    'sprinklr', 'hubspot ai', 'brandwatch',
    'sprout social', 'hootsuite ai',
    'canva ai', 'figma ai', 'adobe ai',
    'loom ai', 'otter.ai', 'rev.ai'
]
