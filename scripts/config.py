import os
from dotenv import load_dotenv

# Load environment variables from .env in the project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Anthropic
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Google Custom Search
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

# ElevenLabs
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM')

# Email
EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT')

# News Agent
TOPIC = os.getenv('NEWS_TOPIC', 'AI in Media')
FREQUENCY_HOURS = int(os.getenv('NEWS_FREQUENCY_HOURS', 1))
NUM_ARTICLES = int(os.getenv('NEWS_NUM_ARTICLES', 5)) 