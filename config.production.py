# Production Configuration
import os

class ProductionConfig:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-super-secret-production-key-change-this')
    DEBUG = False
    TESTING = False
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///instance/media_analysis.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', './backend/training/models/deployment/model/')
    HUGGINGFACE_MODEL = os.environ.get('HUGGINGFACE_MODEL', 'paulmcnally/highlander-ai-model')
    
    # OpenAI Configuration (fallback)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', '/var/log/mediamap/app.log')
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = os.environ.get('RATELIMIT_STORAGE_URL', 'memory://')
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per minute')
    RATELIMIT_HEADERS_ENABLED = True 