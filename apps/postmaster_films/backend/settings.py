from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./mediamap_postmaster.db"
    
    # Redis for job queue
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Media storage
    MEDIA_ROOT: str = "./_postmaster_media"
    
    # ComfyUI for AnimateDiff/SVD
    COMFYUI_BASE_URL: str = "http://localhost:8188"
    
    # Veo 3 Fast Configuration (supports both Gemini API and Vertex AI)
    USE_VEO: bool = True
    
    # Option 1: Direct Gemini API (after approval)
    GEMINI_API_KEY: str | None = None
    VEO_MODEL_ID: str = "veo-3.0-fast-generate-preview"
    
    # Option 2: Vertex AI (enterprise approach)
    GOOGLE_CLOUD_PROJECT: str | None = None
    GOOGLE_CLOUD_REGION: str = "us-central1"
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None
    
    # Veo Generation Settings
    VEO_MAX_DURATION: int = 8  # seconds
    VEO_ASPECT_RATIO: str = "16:9"
    
    # ElevenLabs TTS
    USE_TTS: bool = False
    ELEVENLABS_API_KEY: str | None = None
    ELEVENLABS_VOICE_ID: str = "Rachel"
    
    # Budget management
    EPISODE_VEO_BUDGET_USD: float = 50.0
    VEO_PRICE_PER_SEC: float = 0.40
    
    # General
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()
