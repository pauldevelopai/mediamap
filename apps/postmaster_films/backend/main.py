"""Postmaster Films - FastAPI Main Application"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from .db import Base, engine
from .routers import projects, episodes, scenes, jobs, assets
from .settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Postmaster Films - AI TV Studio",
    description="End-to-end video production pipeline with AI models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Failed to create database tables: {e}")

# Include routers
app.include_router(projects.router, prefix="/projects", tags=["projects"])
app.include_router(episodes.router, prefix="/episodes", tags=["episodes"])
app.include_router(scenes.router, prefix="/scenes", tags=["scenes"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(assets.router, prefix="/assets", tags=["assets"])

@app.get("/")
def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Postmaster Films - AI TV Studio",
        "version": "1.0.0",
        "docs": "/docs",
        "features": [
            "Script to video generation",
            "Budget-based model routing",
            "Veo 3 Fast + AnimateDiff/SVD",
            "Asset management & reuse",
            "Job queue processing",
            "Video assembly & post-production"
        ]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected",
        "media_root": settings.MEDIA_ROOT,
        "veo_enabled": settings.USE_VEO,
        "tts_enabled": settings.USE_TTS
    }

@app.get("/config")
def get_config():
    """Get non-sensitive configuration information"""
    return {
        "veo_enabled": settings.USE_VEO,
        "tts_enabled": settings.USE_TTS,
        "episode_budget_usd": settings.EPISODE_VEO_BUDGET_USD,
        "veo_price_per_sec": settings.VEO_PRICE_PER_SEC,
        "comfyui_url": settings.COMFYUI_BASE_URL,
        "media_root": settings.MEDIA_ROOT
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return HTTPException(status_code=404, detail="Resource not found")

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )

